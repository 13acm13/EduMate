import os
from typing import List
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph,START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import Tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import AnyMessage, add_messages
from uuid import uuid4

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model="llama3-70b-8192",groq_api_key=groq_api_key)

#Loading the pdf + splitting + embedding
loader = PyMuPDFLoader('C:\\braindedmemory\\CP\\GITHUB\\sarvam.ai\\iesc111.pdf')
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(pages)

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

chunk_texts = list(map(lambda d: d.page_content, chunks))
embeddings = bge_embeddings.embed_documents(chunk_texts)
text_embedding_pairs = zip(chunk_texts, embeddings)
db = FAISS.from_embeddings(text_embedding_pairs, bge_embeddings)



#Funtion decleration for the tools - 1)Rag tool 2)QA Test tool
def Textbook_Rag(query: str) -> str:
    llm2 = ChatGroq(model="llama3-8b-8192",groq_api_key=groq_api_key)
    template = """
    You are an educational assistant. Your goal is to help students understand concepts from their textbooks. Use the provided textbook content to explain the following question in detail.

    - Use all relevant context provided to formulate your response.
    - Break down the explanation into key concepts, and explain them clearly.
    - Ensure the answer is helpful for students to learn and understand the topic thoroughly.
    - If the question is unclear or lacks sufficient context, say that you require more detail.
    - If the answer is not in the provided content, say "I don't know."
    - Conclude the answer by summarizing the main points to reinforce understanding.

    {context}

    Question: {question}

    Detailed Explanation:
    """
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = RetrievalQA.from_chain_type(
        llm2,
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": custom_rag_prompt}
    )
    result = rag_chain({"query": query})
    return result["result"]

def Test_Knowledge(topic: str) -> str:
   llm2 = ChatGroq(model="llama3-8b-8192",groq_api_key=groq_api_key)
   textbook_content = Textbook_Rag("Provide comprehensive information about {topic}")
   template = """
    You are an advanced educational assistant specializing in knowledge assessment. Your task is to create a comprehensive test to evaluate students' understanding of {topic}. Use the provided textbook content to formulate relevant and challenging questions.

    Textbook Content:
    {textbook_content}

    Guidelines for Question Generation:
    1. Create a balanced set of questions covering various aspects of the topic.
    2. Ensure questions are clear, unambiguous, and directly related to the textbook content.
    3. Vary the difficulty level to assess different levels of understanding.
    4. For calculation-based questions, provide all necessary information.

    Question Types and Format:
    1. 2-Marker Questions (2 total):
       - 2 Short answer questions
          Format: Q: [Question]
                  A: [Brief answer, 1-2 sentences]

    2. 5-Marker Questions (1 total):
       - Require answers of 100-150 words with significant detail
       Format for each:
       Q: [Question]
       Expected Answer: [Bullet points outlining key elements to be included in the answer]

    3. 10-Marker Questions (1 total):
       - Require answers of 150-200 words with comprehensive detail
       Format for each:
       Q: [Question]
       Expected Answer: [Detailed bullet points covering main concepts, applications, and examples]

    Additional Instructions:
    - Ensure questions progress from fundamental concepts to more advanced applications.
    - Include at least one question that requires critical thinking or problem-solving skills.
    - For 5 and 10-marker questions, include a mix of descriptive, analytical, and application-based questions.
    - Provide clear, concise answers or answer outlines for all questions.

    Your output should be well-formatted, clearly separating each question type and individual questions. 
    """
   custom_rag_prompt = PromptTemplate.from_template(template)
   test_prompt = custom_rag_prompt.format(topic=topic, textbook_content=textbook_content)
   test = llm2.invoke(test_prompt)
    
   return test.content


# Helper functions and helper classes
class State(TypedDict):
    """
    TypedDict for maintaining the state of the agent.
    
    Attributes:
        messages (Annotated[list[AnyMessage], add_messages]): List of messages to be processed.
        user_info (str): Information about the user.
    """
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            state = {**state}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    
def handle_tool_error(state) -> dict:
    """
    Handle errors that occur during tool execution.

    Args:
        state (dict): The current state containing messages and errors.

    Returns:
        dict: Updated state with error messages.
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)} please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Create a ToolNode with fallbacks for error handling.

    Args:
        tools (list): List of tools to be included in the node.

    Returns:
        dict: ToolNode with fallbacks.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            print(msg_repr)
            _printed.add(message.id)


#Tool Declaration
tools = [
    Tool(
        name="Textbook_Rag",
        func=Textbook_Rag,
        description="Useful for answering questions about the textbook content."
    ),
    Tool(
        name="Test_Knowledge",
        func=Test_Knowledge,
        description="Generates various types of questions to test knowledge on a given topic."
    )
]

#Main Prompt that acts as a supervisor for the tools and handles the control flow of the system
react_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an advanced AI educational assistant, specializing in helping students learn from textbooks and improve their study skills. Follow these guidelines strictly:

        Available Tools:
        1. Textbook_Rag(query): Searches the textbook content using Retrieval-Augmented Generation. Use this to answer questions about specific textbook content.
        2. Test_Knowledge(topic): Generates various types of questions to test knowledge on a given topic.

        Process:
        1. Carefully analyze the user's input to determine the appropriate action:
           - Important: For greetings, general questions, or queries not related to the textbook content, respond directly without using any tools.
           - For questions about specific textbook content, use the Textbook_Rag(query) tool.
           - If the user requests a knowledge test or question generation on a topic, use the Test_Knowledge(topic) tool.

        2. Invoke tools only when necessary - Tool Usage Guidelines:
           - Textbook_Rag(query): Use for specific questions about textbook content. Example: "What is sound?"
           - Test_Knowledge(topic): Use when asked to create test questions or assess knowledge on a topic. Example: "Can you create a test on sound?" or "I want to test my knowledge on the topic of sound."
           - Do not use tools for general conversation, greetings, or non-academic queries.

        3. When a tool is invoked, output the tool results as follows:
           - Use the complete tool message and produce a response as the "Final Answer:" by reproducing the tool message.
           - Always print the tool results as the final answer, directly following the tool output. Example: If the tool generates a set of test questions, your response should be "Final Answer: [Tool Output]."
           - Important: Don't summarize or alter the tool output message.
           - Do not add any additional explanation or context beyond what the tool provides.

        4. If you're uncertain about any information, express that uncertainty rather than guessing.

        Remember:
        - Only use tools when absolutely necessary for addressing the user's query.
        - For general conversation or non-textbook related questions, respond directly without tool use.
        - Important: Always formulate the final answer based on the tool's response. If the entire tool response is relevant, include it entirely in the final answer.

        Current time: {time}.
        """
    ),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now())

react_runnable = react_prompt | llm.bind_tools(tools)


builder = StateGraph(State)

# Defining nodes
builder.add_node("assistant", Assistant(react_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))
# Defining edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer to let the graph state persist 
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

def run_graph(question: str):
    _printed = set()
    thread_id = str(uuid4())
    config = {
        "configurable": {
            "user_id": "123",
            "thread_id": thread_id,
        }
    }

    # Stream events from the graph
    events = graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )

    final_message=""
    # Print each event as it occurs
    for event in events:
        # _print_event(event, _printed)
        current_state = event.get("dialog_state")
        if current_state:
            print("Currently in: ", current_state[-1])
        message = event.get("messages")
        if message:
            if isinstance(message, list):
                message = message[-1]
            if message.id not in _printed:
                # Collect the AI message, assuming the last one is the final answer
                final_message = message.content
                msg_repr = message.pretty_repr(html=True)
                print(msg_repr)
                _printed.add(message.id)
    
    if "Final Answer:" in final_message:
        final_message = final_message.split("Final Answer:")[-1].strip()

    return final_message



    
