import json
import pandas as pd
import streamlit as st
from docx import Document
from collections import OrderedDict
from huggingface_hub import InferenceClient
import re

# Initialize the Inference Client
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token="enter your token here",
)


# Query the Model
def query(client, payload):
    response = ""
    for message in client.chat_completion(
        messages=[{"role": "user", "content": payload["inputs"]}],
        max_tokens=payload["parameters"]["max_length"],
        stream=True,
    ):
        response += message.choices[0].delta.content
    print(response)
    return response


# Parse Generated Text to JSON
def parse_generated_text_to_json(text):
    conditions = OrderedDict()
    current_section = None
    current_subsection = None

    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line[0].isdigit():
            parts = line.split(".", 1)
            current_section = parts[0].strip() + "."
            conditions[current_section] = {"title": parts[1].strip(), "points": []}
            current_subsection = None
        elif line.startswith("-"):
            if current_section:
                parts = line.split("-", 1)
                conditions[current_section]["points"].append(parts[1].strip())
        elif line[0].isalpha() and current_section:
            parts = line.split(":", 1)
            if len(parts) == 2:
                current_subsection = parts[0].strip()
                if "subsections" not in conditions[current_section]:
                    conditions[current_section]["subsections"] = OrderedDict()
                conditions[current_section]["subsections"][current_subsection] = {
                    "title": parts[1].strip(),
                    "points": [],
                }
            elif current_subsection:
                conditions[current_section]["subsections"][current_subsection][
                    "points"
                ].append(line.strip())
        elif current_subsection:
            conditions[current_section]["subsections"][current_subsection][
                "points"
            ].append(line.strip())

    return conditions


# Extract Contract Conditions
def extract_conditions(contract_text):
    if not contract_text:
        return {"error": "Empty contract text"}

    prompt = (
        f"Extract all contract conditions from the following text:\n\n{contract_text}"
    )

    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 4096 * 2, "min_length": 30, "truncation": True},
    }
    response_text = query(client, payload)

    if not response_text:
        return {"error": "Empty output from the model"}

    conditions = parse_generated_text_to_json(response_text)

    return conditions


# Format Conditions for Display
def format_conditions_for_display(conditions):
    formatted_conditions = OrderedDict()
    for section, terms in conditions.items():
        formatted_section = OrderedDict()
        formatted_section["title"] = terms["title"]
        formatted_section["points"] = terms["points"]
        if "subsections" in terms:
            formatted_section["subsections"] = OrderedDict()
            for subsection, subterms in terms["subsections"].items():
                formatted_subsection = OrderedDict()
                formatted_subsection["title"] = subterms["title"]
                formatted_subsection["points"] = subterms["points"]
                formatted_section["subsections"][subsection] = formatted_subsection
        formatted_conditions[section] = formatted_section
    return formatted_conditions


# Extract Budget Limit from Conditions
def extract_budget_limit(conditions):
    budget_limit = None
    for section, terms in conditions.items():
        for point in terms["points"]:
            match = re.search(r"\$\d+(\.\d{1,2})?", point)
            if match:
                budget_limit = float(match.group().replace("$", "").replace(",", ""))
                return budget_limit
        if "subsections" in terms:
            for subsection, subterms in terms["subsections"].items():
                for subpoint in subterms["points"]:
                    match = re.search(r"\$\d+(\.\d{1,2})?", subpoint)
                    if match:
                        budget_limit = float(
                            match.group().replace("$", "").replace(",", "")
                        )
                        return budget_limit
    return budget_limit


# Extract Cost from Task Description
def extract_cost(task_description):
    match = re.search(r"\$\d+(\.\d{1,2})?", task_description)
    if match:
        return float(match.group().replace("$", "").replace(",", ""))
    return None


# Analyze Task Description for Compliance
def analyze_task_description(task_description, contract_conditions, contract_text):
    prompt = (
        f"Task Description: {task_description}\n"
        f"Contract Conditions: {json.dumps(contract_conditions, indent=2)}\n"
        f"Contract Text: {contract_text}\n\n"
        "Based on the provided contract conditions and contract text, is the task description compliant?\n"
        "Please start your response with 'Yes' if it is compliant, 'No' if it is not compliant, "
        "or 'Unknown' if the contract conditions do not cover the task description, followed by a brief justification."
    )

    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 500, "min_length": 30, "truncation": True},
    }
    response_text = query(client, payload)

    compliance = "Unknown"
    reason = "The contract conditions do not mention anything about this specific task."

    if response_text:
        lines = response_text.strip().split("\n")
        if lines[0].strip().lower().startswith("yes"):
            compliance = "Yes"
        elif lines[0].strip().lower().startswith("no"):
            compliance = "No"
        elif lines[0].strip().lower().startswith("unknown"):
            compliance = "Unknown"
        if len(lines) > 1:
            reason = lines[1].strip()

    return compliance, reason


# Analyze All Tasks for Compliance
def analyze_tasks(contract_conditions, contract_text, tasks_df):
    analysis_results = []
    for index, row in tasks_df.iterrows():
        task_description = row["Task Description"]
        amount = row["Amount"]

        compliance, reason = analyze_task_description(
            task_description, contract_conditions, contract_text
        )

        analysis_results.append(
            {
                "Task Description": task_description,
                "Amount": amount,
                "Compliance": compliance,
                "Reason": reason,
            }
        )

    return pd.DataFrame(analysis_results)


# Read DOCX File
def read_docx(file):
    document = Document(file)
    full_text = [para.text for para in document.paragraphs]
    return "\n".join(full_text)


# Streamlit App Layout
st.title("Contract Conditions Extraction and Verification")

# Upload Contract File
contract_file = st.file_uploader("Upload Contract", type=["docx"])

# Upload Task Descriptions File
tasks_file = st.file_uploader("Upload Task Descriptions", type="csv")

if contract_file:
    try:
        contract_text = read_docx(contract_file)
    except Exception as e:
        st.error(f"Error reading the DOCX file: {e}")
        st.stop()

    if not contract_text.strip():
        st.error("The uploaded contract file is empty.")
        st.stop()

    # Step 1: Extract contract conditions into JSON
    contract_conditions = extract_conditions(contract_text)
    if "error" in contract_conditions:
        st.error(contract_conditions["error"])
    else:
        formatted_conditions = format_conditions_for_display(contract_conditions)
        st.write("Extracted Contract Conditions:", formatted_conditions)

        # Convert contract conditions to JSON string for download
        contract_conditions_json = json.dumps(formatted_conditions, indent=2)

        st.download_button(
            label="Download Contract Conditions as JSON",
            data=contract_conditions_json,
            file_name="contract_conditions.json",
            mime="application/json",
        )

    if tasks_file:
        tasks_df = pd.read_csv(tasks_file)

        # Step 2: Use the extracted contract conditions to test each Task Description
        analysis_results_df = analyze_tasks(
            contract_conditions, contract_text, tasks_df
        )
        st.write("Task Analysis Results")
        st.dataframe(analysis_results_df)

        st.download_button(
            label="Download Analysis Results",
            data=analysis_results_df.to_csv(index=False).encode("utf-8"),
            file_name="analysis_results.csv",
            mime="text/csv",
        )
