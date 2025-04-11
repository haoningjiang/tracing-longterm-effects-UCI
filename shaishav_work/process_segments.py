import pandas as pd
import csv
from typing import List
from pydantic import BaseModel, Field
import os
import glob
from openai import OpenAI
import instructor
from instructor import Mode

class AnnotationResult(BaseModel):
    """Pydantic model for the annotation result"""
    label: int = Field(
        ..., 
        description="Classification: Label as 1: If the speaker's efforts include a concrete action involving with an external organization (with the organization’s name explicitly provided) explicitly catering to the TOPIC. Label as 0: If there is no explicit mention of an external organization or if the described efforts do not clearly caters to the topic. Be strict on this labelling.",
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
        description="Very Detailed reasoning behind the assigned label based on the coding lines"
    )
    evidence: str = Field(
        ..., 
        description="Lines from the coding section where the Speaker describes their efforts with external collaborators or does not mention them. This should be a direct quote from the coding lines."
    )

def build_prompt(coding_texts, reference_texts):
    """
    Build the annotation prompt for the LLM, including reference context
    plus all coding texts together.
    """
    base_prompt = """
TOPIC: Evaluation and adaptation - The iterative processes of assessing collaborative efforts and integrating feedback to improve strategies, practices, and outcomes. Evaluation and adaptation involve learning from experiences, sharing knowledge, and adjusting workflows or behaviors to align with evolving objectives and challenges.

TASK: Follow these steps one by one:

Reference Context:
Read and understand the provided [REFERENCE] lines to gain context about the situation, the interviewees, and the topic.
Note: Do not use the [REFERENCE] lines for labeling the coding lines.

Analyze the Coding Lines:
Carefully review the [CODING] lines.
Identify and extract descriptions of concrete efforts where the interviewees mention working with external organizations.
Important: There must be an explicit mention of the external organization’s name. Generic phrases like "let's work together" or "we are working with them" do not qualify.

Assess Alignment with the Topic:
Determine if the identified efforts explicitly caters to the TOPIC.


IMPORTANT:
Base your classification solely on the specific efforts and actions related to external collaboration with organizations, not on the general conversational context.
Be strict in applying the criteria: only label as 1 when there is a clear, explicit mention of the TOPIC in the efforts when working with an external organization (with the organization’s name).
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

def process_csv_file(file_path, client):
    """Process a single CSV file and return the results"""
    print(f"Processing file: {file_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
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
            # Filter out rows where the speaker is Seth or Maggie (interviewers)
            coding_rows = group_df[
                (group_df["Segment Type"].str.strip().str.lower() == "coding") &
                (~group_df["Speaker"].str.contains("seth|maggie", case=False, na=False))
            ]

            
            # Skip if there are no coding rows
            if coding_rows.empty:
                coding_texts = ["No coding lines available, label as 0"]
            else:
                coding_texts = [f"{row['Speaker']}: {row['Utterance']}" for _, row in coding_rows.iterrows()]
                
            # Collect all reference utterances as context
            reference_texts = [f"{row['Speaker']}: {row['Utterance']}" for _, row in reference_rows.iterrows()]
            
            # Collect all coding utterances together
            
            
            # # Skip if there's no actual content
            # if not coding_texts:
            #     continue
                
            # Build the prompt with all coding texts together and reference texts
            prompt = build_prompt(coding_texts, reference_texts)
            # Extract segment number from file path
            file_name = os.path.basename(file_path)
            segment_number = None
            if "_Segment" in file_name:
                segment_number = file_name.split("_Segment")[1].split(".")[0]
            
            # Use instructor with OpenAI-compatible client to get a response
            try:
                print(f"Processing segment from {source_value}...")
                
                # Create a message for the LLM
                messages = [
                    {"role": "system", "content": "You are an expert annotator specializing in analyzing whether the given [CODING] lines describe the interviewees' efforts with external collaborators, addressing the specific topic."},
                    {"role": "user", "content": prompt}
                ]
                
                # Use instructor to get structured output
                response = client.chat.completions.create(
                    model="llama3.1",
                    messages=messages,
                    response_model=AnnotationResult
                    , # Set temperature to 0 for deterministic output
                    temperature=0.1,
                )
                
                # Add metadata to the result

                
                result = {
                    "Segment": segment_number if segment_number else os.path.basename(file_path),
                    "file_origin": source_value,
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
                    "Segment": segment_number if segment_number else os.path.basename(file_path),
                    "file_origin": source_value,
                    "label": None,
                    "confidence_score": None,
                    "justification": f"Error: {str(e)}",
                    "evidence": f"Error: {str(e)}"
                })
        
        return file_results
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return [{
            "file_path": str(file_path),
            "file_origin": "Error",
            "label": None,
            "confidence_score": None,
            "justification": f"Error processing file: {str(e)}",
            "evidence": f"Error processing file: {str(e)}"
        }]

def main():
    # Directory containing CSV files
    directory_path = "WY_Segments/WY.01-29-2025" # Change this to your directory path
    
    # Create full path
    full_path = os.path.join(os.getcwd(), directory_path)
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(full_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {full_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    #Create an instructor-enabled OpenAI client
    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        ),
        mode=instructor.Mode.JSON
    )
    
    all_results = []
    
    # Process each CSV file
    for file_path in csv_files:
        file_results = process_csv_file(file_path, client)
        all_results.extend(file_results)
    
    # Write results to a CSV file
    output_path = "WY.01-29-2025_EnA_annotation_results_1.csv"
    
    if all_results:
        # Get column names from the first result
        fieldnames = list(all_results[0].keys())
        
        with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"Processing complete. Results saved to {output_path}")
    else:
        print("No results to write to CSV.")

if __name__ == "__main__":
    main()