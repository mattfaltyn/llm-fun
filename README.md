## Contract Conditions Extraction and Verification

This Streamlit application is designed to extract contract conditions from a DOCX file and verify if task descriptions from a CSV file comply with those conditions. It utilizes the open-source (MIT) [Zephyr 7B β](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) LLM from Hugging Face to analyze the contract text and determine compliance. 

**NOTE**: This project was completed in around an hour. The NLP, backend, and frontend aspects of `app.py` can be greatly improved. The Zephyr 7B β model is unable to accurately extract 100% of the contract conditions, nor is it able to accurately evaluate the contract's compliance with 100% of the item descriptions. Upgrading to a proprietary LLM such as GPT-4o would solve most of these issues. Please contact Matt for more details.

### Deliverables

1. See [contract_conditions.json](https://github.com/mattfaltyn/llm-fun/blob/main/contract_conditions.json) for the extracted contract conditions from `app.py`.
2. See [output.pdf](https://github.com/mattfaltyn/llm-fun/blob/main/output.pdf) for an example output of `app.py`.


### Features

- **Contract Conditions Extraction**: Extracts conditions from a provided DOCX contract file and structures them in a JSON format.
- **Task Description Analysis**: Analyzes task descriptions from a CSV file for compliance with the extracted contract conditions.
- **Downloadable Results**: Provides the extracted contract conditions and analysis results as downloadable JSON and CSV files.

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/contract-compliance-app.git
    cd contract-compliance-app
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up your Hugging Face API token**:
    Replace the token in the code with your Hugging Face API token.

### Usage

1. **Run the application**:
    ```sh
    streamlit run app.py
    ```

2. **Upload the contract file**:
    - Upload a DOCX file containing the contract text.

3. **Upload the task descriptions file**:
    - Upload a CSV file containing the task descriptions and amounts.

4. **View the results**:
    - The app will display the extracted contract conditions and the compliance analysis of each task description.

5. **Download the results**:
    - Download the extracted contract conditions as a JSON file.
    - Download the task analysis results as a CSV file.

### Code Overview

#### Initialization

```python
# Import necessary libraries
import json  # For handling JSON data
import pandas as pd  # For data manipulation and analysis
import streamlit as st  # For creating web applications
from docx import Document  # For working with Word documents
from collections import OrderedDict  # For maintaining the order of dictionary items
from huggingface_hub import InferenceClient  # For accessing models hosted on Hugging Face Hub
import re  # For regular expression operations

# Initialize the Hugging Face inference client with the specified model and authentication token
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",  # The model to be used for inference
    token="enter your token here",  # Authentication token for accessing the model
)
```

#### Model Query

```python
# Define a function named 'query' to interact with the Hugging Face model using the provided client and payload
def query(client, payload):
    response = ""  # Initialize an empty string to store the response
    
    # Iterate over the stream of messages returned by the chat completion method of the client
    for message in client.chat_completion(
        messages=[{"role": "user", "content": payload["inputs"]}],  # Prepare the input message in the required format
        max_tokens=payload["parameters"]["max_length"],  # Specify the maximum number of tokens for the response
        stream=True,  # Enable streaming of the response
    ):
        # Append the content of the message to the response string
        response += message.choices[0].delta.content
    
    print(response)  # Print the complete response
    return response  # Return the response string
```

#### Parse Generated Text to JSON

```python
# Define a function to parse generated text into a JSON-like structure
def parse_generated_text_to_json(text):
    conditions = OrderedDict()  # Initialize an ordered dictionary to store the parsed data
    current_section = None  # Variable to keep track of the current section
    current_subsection = None  # Variable to keep track of the current subsection

    # Split the input text into lines
    lines = text.split("\n")

    # Iterate over each line in the text
    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
        if not line:  # Skip empty lines
            continue

        # Check if the line represents a new section (starts with a digit)
        if line[0].isdigit():
            parts = line.split(".", 1)  # Split the line at the first dot
            current_section = parts[0].strip() + "."  # Extract the section number
            conditions[current_section] = {"title": parts[1].strip(), "points": []}  # Add the section to the conditions
            current_subsection = None  # Reset the current subsection
        # Check if the line represents a bullet point (starts with a hyphen)
        elif line.startswith("-"):
            if current_section:  # Ensure there is a current section
                parts = line.split("-", 1)  # Split the line at the first hyphen
                conditions[current_section]["points"].append(parts[1].strip())  # Add the point to the section
        # Check if the line represents a new subsection (starts with an alphabet character and there is a current section)
        elif line[0].isalpha() and current_section:
            parts = line.split(":", 1)  # Split the line at the first colon
            if len(parts) == 2:  # Ensure the line has a title part after the colon
                current_subsection = parts[0].strip()  # Extract the subsection title
                # Initialize the subsection within the current section
                if "subsections" not in conditions[current_section]:
                    conditions[current_section]["subsections"] = OrderedDict()
                conditions[current_section]["subsections"][current_subsection] = {
                    "title": parts[1].strip(),
                    "points": [],
                }
            elif current_subsection:  # If no title part, treat the line as a point under the current subsection
                conditions[current_section]["subsections"][current_subsection][
                    "points"
                ].append(line.strip())
        # If the line is part of the current subsection
        elif current_subsection:
            conditions[current_section]["subsections"][current_subsection][
                "points"
            ].append(line.strip())

    return conditions  # Return the parsed conditions as an ordered dictionary
```

#### Extract Contract Conditions

```python
# Define a function to extract conditions from a given contract text
def extract_conditions(contract_text):
    # Check if the contract text is empty
    if not contract_text:
        return {"error": "Empty contract text"}  # Return an error message if the text is empty

    # Create a prompt for the model to extract conditions from the contract text
    prompt = (
        f"Extract all contract conditions from the following text:\n\n{contract_text}"
    )

    # Prepare the payload for the model query
    payload = {
        "inputs": prompt,  # The prompt input for the model
        "parameters": {"max_length": 4096 * 2, "min_length": 30, "truncation": True},  # Parameters for the model
    }
    
    # Call the query function to get the response from the model
    response_text = query(client, payload)

    # Check if the response from the model is empty
    if not response_text:
        return {"error": "Empty output from the model"}  # Return an error message if the output is empty

    # Parse the response text into a JSON-like structure
    conditions = parse_generated_text_to_json(response_text)

    return conditions  # Return the parsed conditions
```

#### Format Conditions for Display

```python
# Define a function to format conditions for display
def format_conditions_for_display(conditions):
    formatted_conditions = OrderedDict()  # Initialize an ordered dictionary to store the formatted conditions
    
    # Iterate over each section in the conditions
    for section, terms in conditions.items():
        formatted_section = OrderedDict()  # Initialize an ordered dictionary for the current section
        formatted_section["title"] = terms["title"]  # Add the title of the section
        formatted_section["points"] = terms["points"]  # Add the points of the section
        
        # Check if there are subsections in the current section
        if "subsections" in terms:
            formatted_section["subsections"] = OrderedDict()  # Initialize an ordered dictionary for the subsections
            # Iterate over each subsection in the current section
            for subsection, subterms in terms["subsections"].items():
                formatted_subsection = OrderedDict()  # Initialize an ordered dictionary for the current subsection
                formatted_subsection["title"] = subterms["title"]  # Add the title of the subsection
                formatted_subsection["points"] = subterms["points"]  # Add the points of the subsection
                # Add the formatted subsection to the current section's subsections
                formatted_section["subsections"][subsection] = formatted_subsection
        
        # Add the formatted section to the formatted conditions
        formatted_conditions[section] = formatted_section
    
    return formatted_conditions  # Return the formatted conditions
```

#### Extract Budget Limit from Conditions

```python
# Define a function to extract the budget limit from the conditions
def extract_budget_limit(conditions):
    budget_limit = None  # Initialize the budget limit variable to None
    
    # Iterate over each section in the conditions
    for section, terms in conditions.items():
        # Iterate over each point in the current section
        for point in terms["points"]:
            # Search for a monetary value in the point
            match = re.search(r"\$\d+(\.\d{1,2})?", point)
            if match:  # If a match is found
                # Convert the matched value to a float and assign it to budget_limit
                budget_limit = float(match.group().replace("$", "").replace(",", ""))
                return budget_limit  # Return the budget limit
        
        # Check if there are subsections in the current section
        if "subsections" in terms:
            # Iterate over each subsection in the current section
            for subsection, subterms in terms["subsections"].items():
                # Iterate over each point in the current subsection
                for subpoint in subterms["points"]:
                    # Search for a monetary value in the subpoint
                    match = re.search(r"\$\d+(\.\d{1,2})?", subpoint)
                    if match:  # If a match is found
                        # Convert the matched value to a float and assign it to budget_limit
                        budget_limit = float(
                            match.group().replace("$", "").replace(",", "")
                        )
                        return budget_limit  # Return the budget limit
    
    return budget_limit  # Return the budget limit if found, otherwise None
```

#### Extract Cost from Task Description

```python
# Define a function to extract the cost from a given task description
def extract_cost(task_description):
    # Search for a monetary value in the task description
    match = re.search(r"\$\d+(\.\d{1,2})?", task_description)
    if match:  # If a match is found
        # Convert the matched value to a float and return it
        return float(match.group().replace("$", "").replace(",", ""))
    return None  # Return None if no monetary value is found
```

#### Analyze Task Description for Compliance

```python
# Define a function to analyze a task description for compliance with contract conditions
def analyze_task_description(task_description, contract_conditions, contract_text):
    # Create a prompt for the model to analyze the task description against the contract conditions and text
    prompt = (
        f"Task Description: {task_description}\n"  # Include the task description in the prompt
        f"Contract Conditions: {json.dumps(contract_conditions, indent=2)}\n"  # Include the formatted contract conditions
        f"Contract Text: {contract_text}\n\n"  # Include the full contract text
        "Based on the provided contract conditions and contract text, is the task description compliant?\n"
        "Please start your response with 'Yes' if it is compliant, 'No' if it is not compliant, "
        "or 'Unknown' if the contract conditions do not cover the task description, followed by a brief justification."
    )

    # Prepare the payload for the model query
    payload = {
        "inputs": prompt,  # The prompt input for the model
        "parameters": {"max_length": 500, "min_length": 30, "truncation": True},  # Parameters for the model
    }
    
    # Call the query function to get the response from the model
    response_text = query(client, payload)

    # Initialize compliance and reason variables with default values
    compliance = "Unknown"
    reason = "The contract conditions do not mention anything about this specific task."

    # If the response from the model is not empty
    if response_text:
        lines = response_text.strip().split("\n")  # Split the response text into lines
        # Check the first line for compliance status
        if lines[0].strip().lower().startswith("yes"):
            compliance = "Yes"
        elif lines[0].strip().lower().startswith("no"):
            compliance = "No"
        elif lines[0].strip().lower().startswith("unknown"):
            compliance = "Unknown"
        # If there is more than one line, use the second line as the reason
        if len(lines) > 1:
            reason = lines[1].strip()

    return compliance, reason  # Return the compliance status and reason
```

#### Analyze All Tasks for Compliance

```python
# Define a function to analyze multiple tasks for compliance with contract conditions
def analyze_tasks(contract_conditions, contract_text, tasks_df):
    analysis_results = []  # Initialize a list to store the analysis results
    
    # Iterate over each row in the tasks DataFrame
    for index, row in tasks_df.iterrows():
        task_description = row["Task Description"]  # Get the task description from the current row
        amount = row["Amount"]  # Get the amount from the current row

        # Analyze the task description for compliance
        compliance, reason = analyze_task_description(
            task_description, contract_conditions, contract_text
        )

        # Append the analysis result to the results list
        analysis_results.append(
            {
                "Task Description": task_description,  # Include the task description in the result
                "Amount": amount,  # Include the amount in the result
                "Compliance": compliance,  # Include the compliance status in the result
                "Reason": reason,  # Include the reason for the compliance status in the result
            }
        )

    # Convert the analysis results list to a DataFrame and return it
    return pd.DataFrame(analysis_results)
```

#### Read DOCX File

```python
# Define a function to read the content of a DOCX file
def read_docx(file):
    document = Document(file)  # Load the DOCX file into a Document object
    # Extract the text from each paragraph in the document and store it in a list
    full_text = [para.text for para in document.paragraphs]
    # Join the list of paragraph texts into a single string with newline characters separating paragraphs
    return "\n".join(full_text)
```

#### Streamlit App Layout

```python
# Set the title of the Streamlit app
st.title("Contract Conditions Extraction and Verification")

# Upload Contract File
contract_file = st.file_uploader("Upload Contract", type=["docx"])

# Upload Task Descriptions File
tasks_file = st.file_uploader("Upload Task Descriptions", type="csv")

if contract_file:
    try:
        # Read the uploaded DOCX contract file
        contract_text = read_docx(contract_file)
    except Exception as e:
        # Display an error message if there is an issue reading the DOCX file
        st.error(f"Error reading the DOCX file: {e}")
        st.stop()

    # Check if the contract text is empty
    if not contract_text.strip():
        st.error("The uploaded contract file is empty.")
        st.stop()

    # Step 1: Extract contract conditions into JSON
    contract_conditions = extract_conditions(contract_text)
    if "error" in contract_conditions:
        # Display an error message if there is an issue extracting conditions
        st.error(contract_conditions["error"])
    else:
        # Format the extracted contract conditions for display
        formatted_conditions = format_conditions_for_display(contract_conditions)
        st.write("Extracted Contract Conditions:", formatted_conditions)

        # Convert the formatted contract conditions to a JSON string for download
        contract_conditions_json = json.dumps(formatted_conditions, indent=2)

        # Provide a download button for the contract conditions JSON
        st.download_button(
            label="Download Contract Conditions as JSON",
            data=contract_conditions_json,
            file_name="contract_conditions.json",
            mime="application/json",
        )

    if tasks_file:
        # Read the uploaded CSV file containing task descriptions
        tasks_df = pd.read_csv(tasks_file)

        # Step 2: Use the extracted contract conditions to test each Task Description
        analysis_results_df = analyze_tasks(contract_conditions, contract_text, tasks_df)
        st.write("Task Analysis Results")
        st.dataframe(analysis_results_df)

        # Provide a download button for the analysis results as a CSV file
        st.download_button(
            label="Download Analysis Results",
            data=analysis_results_df.to_csv(index=False).encode("utf-8"),
            file_name="analysis_results.csv",
            mime="text/csv",
        )
```

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
