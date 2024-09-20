import streamlit as st
from together import Together
import csv
import pandas as pd

# Initialize the client with your API key
client = Together(api_key="")

# Function to get user input for the query
def get_user_query():
    return st.sidebar.text_input("Enter your query:")

# Function to get user-defined documents from file upload
def get_documents_from_file(uploaded_file):
    documents = []
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            documents = df.iloc[:, 0].tolist()
        elif uploaded_file.name.endswith('.txt'):
            documents = [line.decode('utf-8').strip() for line in uploaded_file.read().splitlines()]
    return documents

# Function to get user-defined documents from text inputs
def get_documents_from_text_inputs():
    documents = []
    doc_count = 0
    st.sidebar.write("Enter documents (leave blank to finish):")
    
    while True:
        doc = st.sidebar.text_input(f"Document {doc_count + 1}", key=f"document_{doc_count}")
        if doc == "":
            break
        documents.append(doc)
        doc_count += 1
    return documents

# Function to get user-defined number of top results
def get_top_n():
    return st.sidebar.number_input("Enter the number of top results to return:", min_value=1, value=2)

# Function to get model choice
def get_model_choice():
    models = ["Salesforce/Llama-Rank-V1", "another/model", "yet/another/model"]
    return st.sidebar.selectbox("Select a model:", models)

# Function to save results to a specified file format
def save_results(results, documents, format='txt'):
    filename = f'ranked_results.{format}'
    with open(filename, 'w', newline='') as f:
        if format == 'csv':
            writer = csv.writer(f)
            writer.writerow(['Document Index', 'Document', 'Relevance Score'])
            for result in results:
                writer.writerow([result.index, documents[result.index], f"{result.relevance_score:.4f}"])
        else:
            for result in results:
                f.write(f"Document Index: {result.index}\n")
                f.write(f"Document: {documents[result.index]}\n")
                f.write(f"Relevance Score: {result.relevance_score:.4f}\n\n")

    st.sidebar.success(f"Results saved to '{filename}'.")

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #00b4d8;'>Document Ranking with Together AI</h1>", unsafe_allow_html=True)

# Get user input
query = get_user_query()

# File upload option
uploaded_file = st.sidebar.file_uploader("Upload a CSV or TXT file with documents:", type=["csv", "txt"])

# Get documents either from uploaded file or text inputs
if uploaded_file is not None:
    documents = get_documents_from_file(uploaded_file)
else:
    documents = get_documents_from_text_inputs()

top_n = get_top_n()
model = get_model_choice()

# Ask the user for output format before ranking documents
output_format = st.sidebar.selectbox("Choose output format:", ['txt', 'csv'])

# Use the rerank API to rank documents based on relevance to the query
if st.sidebar.button("Rank Documents"):
    try:
        response = client.rerank.create(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n
        )

        # Print the results in a structured format
        st.subheader("Ranked Results:")
        results_data = []
        for result in response.results:
            results_data.append({
                "Document Index": result.index,
                "Document": documents[result.index],
                "Relevance Score": f"{result.relevance_score:.4f}"
            })
        
        # Display results in a table format
        st.table(results_data)

        # Save results based on the selected output format
        save_results(response.results, documents, format=output_format)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Save search history
history_file = 'search_history.txt'
with open(history_file, 'a') as f:
    f.write(f"Query: {query}\n")
    f.write(f"Documents: {', '.join(documents)}\n")
    f.write(f"Top Results: {top_n}\n")
    f.write(f"Selected Model: {model}\n")
    f.write("\n")
