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
        description="Classification: Label as 1: If the speaker's efforts include a concrete action involving with an external organization explicitly catering to the TOPIC. Label as 0: If there is no  mention of an external organization or if the described efforts do not clearly caters to the topic. Be strict on this labelling.",
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
        description="Very Long Detailed reasoning behind the assigned label, including how the coding lines align with the topic and the specific efforts described."
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
    few_shot_examples = """
Examples to consider for domain-specific annotation, Evaluate and label the coding lines based on the given topic.:

--------------------------------------------------------------------------------
Example A – Positive Case (Label 1)
--------------------------------------------------------------------------------
Transcript Example:
"Now, I am curious, so that the one up in level is more of a drop-in, I think, program for the middle school kids. But we can certainly try and see what kind of response and information we get."

Annotation:
- Label: 1
- Confidence Score: 1.0
- Justification:
  This snippet is classified as positive because it reflects an openness to gather feedback and new insights, which is a core element of knowledge and information sharing. The speaker expresses curiosity about the response they might receive, indicating an active interest in exchanging information and learning from the outcomes. This willingness to adapt based on external input aligns with the criteria of systematically integrating new information to improve practices.
- Evidence:
  "we can certainly try and see what kind of response and information we get"

--------------------------------------------------------------------------------
Example B – Positive Case (Label 1)
--------------------------------------------------------------------------------
Transcript Example:
"I would just offer to Maggie. I went in prepared for this meeting and read the paper on your website related to this project. It helped me make connections between in-school and out-of-school experiences, enriching my understanding of the subject."

Annotation:
- Label: 1
- Confidence Score: 1.0
- Justification:
  In this snippet, the speaker actively engages with shared resources by reading a colleague's paper and using that information to create broader connections. This behavior exemplifies knowledge sharing, as it involves seeking, absorbing, and integrating external insights into one’s own perspective. The process of linking different experiences to enhance understanding is a clear demonstration of collaborative learning and information exchange.
- Evidence:
  "read the paper on your website" and "making connections between in-school and out-of-school experiences"

--------------------------------------------------------------------------------
Example C – Negative Case (Label 0)
--------------------------------------------------------------------------------
Transcript Example:
"We are focusing strictly on our internal processes and reviewing our own data. There is no discussion of external input or sharing of insights beyond our immediate team."

Annotation:
- Label: 0
- Confidence Score: 1.0
- Justification:
  This snippet is classified as negative because it confines its scope to internal review without any indication of seeking or integrating external knowledge. There is no evidence of active information exchange or collaborative learning with other stakeholders. The focus remains inward, which does not satisfy the criteria for systematic knowledge and information sharing.
- Evidence:
  "focusing strictly on our internal processes" and "no discussion of external input or sharing of insights"
--------------------------------------------------------------------------------
"""


    base_prompt = """
TOPIC: Knowledge and information sharing - The systematic exchange and integration of information, expertise, and insights between stakeholders to enhance mutual understanding, innovation, and decision-making. This theme encompasses formal and informal mechanisms for sharing explicit, tacit, and operational knowledge to support collaborative learning and efficiency.

Example: A math teacher shares a fun way to teach fractions with another teacher at a different school, helping more students understand the topic.

TASK: Follow these steps one by one:

Reference Context:
Read and understand the provided [REFERENCE] lines to gain context about the situation, the interviewees, and the topic.
Note: Do not use the [REFERENCE] lines for labeling the coding lines.

Analyze the Coding Lines:
Carefully review the [CODING] lines.
Identify and extract descriptions of concrete efforts where the interviewees mention working with external organizations.
Important: There must be mention of the external organization’s name. 

Assess Alignment with the Topic:
Determine if the identified efforts explicitly caters to the given TOPIC.


IMPORTANT:
Base your classification solely on the specific efforts and actions related to external collaboration with organizations, not on the general conversational context.
Be strict in applying the criteria: only label as 1 when there is a clear mention of the TOPIC in the efforts of the interviewees when working with an external organization.
    """

    # Combine all reference lines into one context block
    reference_section = ""
    if reference_texts:
        reference_section = "\n\n[Reference]:\n" + "\n".join(reference_texts)

    # Combine all coding segments together
    coding_section = "\n\n[Coding]:\n" + "\n".join(coding_texts)

    # Combine everything into one prompt
    complete_prompt = base_prompt.strip() + few_shot_examples + reference_section + coding_section
    return complete_prompt

def process_csv_file(file_path, client, original_label):
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
                    temperature=0.01,
                )
                
                # Add metadata to the result

                
                result = {
                    "Segment": segment_number if segment_number else os.path.basename(file_path),
                    "file_origin": source_value,
                    "label": response.label,
                    "human_label": original_label,
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
                    "human_label": original_label,
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
        
    #Create an instructor-enabled OpenAI client
    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        ),
        mode=instructor.Mode.JSON
    )


    # Two subdirectories are present here, named as 1 and 0
    labels = [0,1]
    all_results = []

    for label in labels:
        #fetch the csv files from the subdirectory named as 1 and 0
        directory_path = f"Samples/KIS/{label}"
        full_path = os.path.join(os.getcwd(), directory_path)
    
        # Get all CSV files in the directory
        csv_files = glob.glob(os.path.join(full_path, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {full_path}")
            return
        
        print(f"Found {len(csv_files)} CSV files to process")

    
        # Process each CSV file
        for file_path in csv_files:
            file_results = process_csv_file(file_path, client, label)
            all_results.extend(file_results)
    
    # Write results to a CSV file
    output_path = "Samples/KIS/results.csv"
    
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