from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.runnables import RunnableSequence
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

set_device = "cpu"

# Create text
textgen_model = pipeline("text-generation", model="zai-org/GLM-4.7", device=set_device)
messages = [
    {"role": "user", "content": "Create a short story abot Batman"},
]
text_to_summarize = textgen_model(messages)

print("\n🔹 **Generated text:**")
print(text_to_summarize)

# Load Hugging Face Summarization Pipeline
sum_model = pipeline("summarization", model="facebook/bart-large-cnn", device=set_device)

# Wrap it inside LangChain
llm = HuggingFacePipeline(pipeline=sum_model)

# Get user input
t_age = 4 # input("Enter target age for simplification:\n")

# Create the prompt template for summarization
chat_promt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text in a way a {age} year old would understand"),
    ("human", "{text}"),
])

messages = chat_promt.format_messages(age = t_age, text = text_to_summarize)

# Execute the summarization chain
summary = llm.invoke(messages)

print("\n🔹 **Generated summary:**")
print(summary)

# Analyze summary
sentiment_pipeline = pipeline("sentiment-analysis", device=set_device)
analysis = sentiment_pipeline(summary)

print("\n🔹 **Generated analysis:**")
print(analysis)

