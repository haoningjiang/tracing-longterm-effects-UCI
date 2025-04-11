import pandas as pd
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import subprocess
import tempfile
import os
import re
import instructor
from instructor import Mode
import glob
from pathlib import Path

class AnnotationResult(BaseModel):
    """Pydantic model for the annotation result"""
    label: int = Field(
        ..., 
        description="1 if the topic is discussed in the coding lines, 0 if not",
        ge=0, 
        le=1
    )
    confidence_score: float = Field(
        ..., 
        description="Confidence level in the label (0.0 to 1.0)",
        ge=0.0, 
        le=1.0
    )
    justification: str = Field(
        ..., 
        description="Reasoning behind the assigned label"
    )
    evidence: List[str] = Field(
        ..., 
        description="Specific quotes from coding segments that support the label"
    )

def build_prompt(coding_texts, reference_texts):
    """
    Build the annotation prompt for the LLM, including reference context
    plus all coding texts together.
    """
    base_prompt = """
ROLE: You are an expert annotator focused on analyzing the content of interview segments from youth participating in creative experiences, doing or making art, or helping others do or make art.

TOPIC: Participating in the arts can foster and influence many types of relationships between people. This is a general category. Many types of relationships can be included in this category. Participants mention or describe accounts about one or more bonds with other people that were fostered by and or influenced by their participation in their arts program or arts practice. These interpersonal relationships can include friendships, mentorships, bonds between classmates and between students and teachers, professional connections, connections that lead to career opportunities, and social networks, among others.

TASK: First, analyze ALL the interview segments labeled [Coding] and determine whether the above TOPIC is discussed in ANY of these segments. Use the [Reference] lines **only** as context to understand the [Coding] lines, but only label based on the [Coding] lines collectively.

Consider all the coding segments together as a single unit and provide ONE overall label.
    """

    # Combine all reference lines into one context block
    reference_section = ""
    if reference_texts:
        reference_section = "\n\n[Reference]:\n" + "\n".join(reference_texts)

    # Combine all coding segments together
    coding_section = "\n\n[Coding]:\n" + "\n".join(coding_texts)

    # Combine everything into one prompt
    complete_prompt = base_prompt.strip() + reference_section + coding_section
    return complete_prompt

class OllamaClient:
    """Custom client for Ollama that works with instructor"""
    
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
    
    def completion(self, messages, response_model=None, **kwargs):
        """Process completion with Ollama CLI and return structured output"""
        # Extract the user's message content
        user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
        if not user_message:
            raise ValueError("No user message found in the messages list")
        
        # Create a temporary file to store the prompt
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as f:
            f.write(user_message)
            temp_file_path = f.name
        
        try:
            # Run ollama with the prompt file as input and add schema instructions
            schema_instructions = ""
            if response_model:
                schema_instructions = f"\n\nYour response MUST be valid JSON that matches this Python class:\n{response_model.__pydantic_core_schema__}"
                
            cmd = ['ollama', 'run', self.model_name, '-f', temp_file_path]
            if schema_instructions:
                # Write schema instructions to another temp file and include in command
                with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as f:
                    f.write(schema_instructions)
                    schema_file_path = f.name
                cmd.extend(['-f', schema_file_path])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            response = result.stdout.strip()
            
            # Extract JSON from the response
            json_match = re.search(r'{[\s\S]*}', response)
            if json_match:
                json_str = json_match.group(0)
                json_data = json.loads(json_str)
                
                # If we have a response model, validate against it
                if response_model:
                    return response_model(**json_data)
                return json_data
            else:
                raise ValueError("No valid JSON found in response")
        
        finally:
            # Clean up temp files
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if 'schema_file_path' in locals() and os.path.exists(schema_file_path):
                os.remove(schema_file_path)

def process_excel_file(file_path, client):
    """Process a single Excel file and return the results"""
    print(f"Processing file: {file_path}")
    
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        # Check if required columns exist
        required_columns = ["Speaker", "Utterance", "File Origin", "Segment Type"]
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in {file_path}")
                return []
        
        # Group by 'File Origin' so that reference and coding lines from the same transcript are together
        grouped = df.groupby("File Origin")
        
        file_results = []
        
        # Iterate over each group (transcript)
        for source_value, group_df in grouped:
            # Skip if source is NaN
            if pd.isna(source_value):
                continue
                
            # Separate reference rows from coding rows
            reference_rows = group_df[group_df["Segment Type"].str.strip().str.lower() == "reference"]
            coding_rows = group_df[group_df["Segment Type"].str.strip().str.lower() == "coding"]
            
            # Skip if there are no coding rows
            if coding_rows.empty:
                continue
                
            # Collect all reference utterances as context
            reference_texts = [f"{row['Speaker']}: {row['Utterance']}" for _, row in reference_rows.iterrows()]
            
            # Collect all coding utterances together
            coding_texts = [f"{row['Speaker']}: {row['Utterance']}" for _, row in coding_rows.iterrows()]
            
            # Skip if there's no actual content
            if not coding_texts:
                continue
                
            # Build the prompt with all coding texts together and reference texts
            prompt = build_prompt(coding_texts, reference_texts)
            
            # Use instructor with Ollama client to get a response
            try:
                print(f"Processing segment from {source_value}...")
                
                # Create a message for the LLM
                messages = [
                    {"role": "system", "content": "You are an expert annotator analyzing interview transcripts."},
                    {"role": "user", "content": prompt}
                ]
                
                # Use instructor to enforce the schema
                response = client.completion(
                    messages=messages,
                    response_model=AnnotationResult
                )
                
                # Add metadata to the result
                result = {
                    "file_path": str(file_path),
                    "file_origin": source_value,
                    "prompt": prompt,
                    "label": response.label,
                    "confidence_score": response.confidence_score,
                    "justification": response.justification,
                    "evidence": response.evidence
                }
                
                file_results.append(result)
                print(f"Successfully processed segment from {source_value}")
                
            except Exception as e:
                print(f"Error processing {source_value}: {str(e)}")
                # Add error information to results
                file_results.append({
                    "file_path": str(file_path),
                    "file_origin": source_value,
                    "error": str(e)
                })
        
        return file_results
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return [{
            "file_path": str(file_path),
            "error": str(e)
        }]

def main():
    # Directory containing Excel files
    directory_path = "WY_Segments/WY.01-29-2025"
    
    # Create full path
    full_path = os.path.join(os.getcwd(), directory_path)
    
    # Get all Excel files in the directory
    excel_files = glob.glob(os.path.join(full_path, "*.xlsx"))
    excel_files.extend(glob.glob(os.path.join(full_path, "*.xls")))
    
    if not excel_files:
        print(f"No Excel files found in {full_path}")
        return
    
    print(f"Found {len(excel_files)} Excel files to process")
    
    # Create an instructor client with our custom Ollama client
    ollama_client = OllamaClient(model_name="llama3.1")
    client = instructor.from_(
        ollama_client,
        mode=Mode.JSON
    )
    
    all_results = []
    
    # Process each Excel file
    for file_path in excel_files:
        file_results = process_excel_file(file_path, client)
        all_results.extend(file_results)
    
    # Write results to a JSON file
    output_path = "annotation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Processing complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()