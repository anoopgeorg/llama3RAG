from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph

from src.configurations.config import ConfigManager
from src.GraphState import GraphState
from src.vault import Vault


class Prompter:
    def __init__(self, llm):
        # Add all the prompts from some config file
        self.llm = llm
        config_mgr = ConfigManager()
        self.config = config_mgr.getPrompterConfig()
        self.simple_template = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful and wise AI assistant called Alfy.
        Answer "Don't know" when you do not know the answer to a question.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>Give your best response for {question}<|eot_id|>"""

        self.doc_context_template = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful and wise AI assistant called Alfy.
        Answer "Don't know" when you do not know the answer to a question.<|eot_id|>
        
        <|start_header_id|>user<|end_header_id|>Give your best response for {input}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>Context:{context}<|eot_id|>
        """
        self.retrieval_grader = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        self.router_template = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a expert router. You are going to asses the question 
        by validating against the vector store context, If there is a relation between question and vector store context, then respond with a binary choice
        "vector_store" else "web_search". 
        Return a JSON with a single key 'route'.
        Question to route: {question}
        Vector store context: {vector_store_context}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        """

        self.generate_template = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at answering questions. Use the context provided
        to derive the answer to the user question. If you don't know the answer just say that you don't know. Use a maximum of five sentences 
        and keep the answer concise.<|eot_id|><|start_header_id|>user<|end_header_id|>
        User Question: {question}
        Context: {context}
        Answer: <|eot_id|> <|start_header_id|>assistant<|end_header_id|>
        """

    def simplePrompt(self, query):
        llm = ChatOllama(model="llama3", temperature=0.3)
        simple_prompt = PromptTemplate(
            template=self.simple_template, input_variables=["question"]
        )
        simple_chain = LLMChain(llm=llm, prompt=simple_prompt)
        response = simple_chain.invoke(input={"question": query})
        print(type(response))
        print(response)
        return response

    def documentPrompt(self, query: str):
        llm = ChatOllama(model="llama3", temperature=0)
        vault = Vault()
        retriever = vault.getRetriever()
        print("Entered document prompt mode")
        raw_prompt = PromptTemplate(
            template=self.doc_context_template, input_variables=["question", "context"]
        )
        # print(raw_prompt.format(question=query, context="testes"))
        doc_chain = create_stuff_documents_chain(llm=llm, prompt=raw_prompt)
        # doc_chain.invoke(input={"question": query})
        retriever_chain = create_retrieval_chain(
            retriever=retriever, combine_docs_chain=doc_chain
        )
        result = retriever_chain.invoke(input={"input": query})
        # return result
        print(result)
        sources = []
        for doc in result["context"]:
            sources.append(
                {
                    "sources": doc.metadata["source"],
                    "page_content": doc.page_content,
                    "page_no": doc.metadata["page"],
                }
            )

        response_answer = {"answer": result["answer"], "source": sources}
        print(response_answer)
        return response_answer

    def routingPrompt(self, state: GraphState):
        print("---ROUTING---")
        llm = ChatOllama(model="llama3", format="json", temperature=0)
        query = state["query"]
        route_prompt = PromptTemplate(
            template=self.router_template,
            input_variables=["question", "vector_store_context"],
        )

        # for now the vector_store_context is going to be the file names in knowledge bank
        knowledge_dir = self.config.knowledge_dir
        knowledge_context = ",".join(
            [str(f.name.split(".")[0]) for f in knowledge_dir.glob("*")]
        )
        question_router = route_prompt | llm | JsonOutputParser()
        result = question_router.invoke(
            input={"question": query, "vector_store_context": knowledge_context}
        )
        route = result.get("route")
        print(f"---ROUTED TO {str(route).upper()} ---")
        return route
        # return {"route": route, "question": query}

    def retrieve(self, state: GraphState):
        print("---RETRIEVE---")
        question = state["query"]
        vault = Vault()
        retriever = vault.getRetriever()
        docs = retriever.invoke(question)
        print(f"Retrieved docs : {len(docs)}")
        print(f"{type(docs)}")
        print("---RETRIEVE COMPLETE---")
        return {"context": docs, "question": question, "route": "vector_store"}

    def web_search(self, state: GraphState):
        print("---WEB SEARCH---")
        question = state["query"]
        print(question)
        # documents = state["context"]
        web_context = state["web_context"]
        web_search_tool = TavilySearchResults(k=3)
        web_result = web_search_tool.invoke({"query": question})

        if web_context is not None:
            web_context.append(web_result)
        else:
            web_context = [web_result]

        return {"web_context": web_context, "question": question, "route": "web_search"}

    def decideToGenerate(self, state: GraphState):
        print("---ASSES GRADE FOR DOCUMENT RELEVANCE---")
        web_search_flag = state["web_search_flag"]

        if web_search_flag == True:
            print("---DOCUMENT RELEVANCE ASSESSMENT FAILED WEB SEARCH SET---")
            return "web_search"
        else:
            print("---DOCUMENT RELEVANCE ASSESSMENT PASSED---")
            return "generate"

    def gradeContext(self, state: GraphState):
        print("---GRADE CONTEXT RELEVANCE---")
        question = state["query"]
        context = state["context"]
        # web_context = state["web_context"]
        llm = ChatOllama(model="llama3", temperature=0, format="json")
        grade_prompt = PromptTemplate(
            template=self.retrieval_grader, input_variables=["question", "document"]
        )
        grader = grade_prompt | llm | JsonOutputParser()
        filtered_docs = []
        # Even if a single document is not relevant we will push for web search
        web_search_flag = False if len(context) > 0 else True
        for doc in context:
            doc_result = (
                grader.invoke({"question": question, "document": doc})
                .get("score")
                .lower()
            )
            if doc_result == "yes":
                filtered_docs.append(doc)
                print("---DOC IS RELEVANT---")
            else:
                print("---DOC IS NOT RELEVANT, WEB SEARCH FLAG SET---")
                web_search_flag = True
        return {"context": filtered_docs, "web_search_flag": web_search_flag}

    def generate(self, state: GraphState):
        print("---GENERATE---")
        question = state["query"]
        print(question)
        context = state["context"]
        web_context = state["web_context"]
        web_search_flag = state["web_search_flag"]
        relevant_context = context if web_search_flag == False else web_context
        llm = ChatOllama(model="llama3", temperature=0)
        generate_prompt = PromptTemplate(
            template=self.generate_template, input_variables=["question", "context"]
        )
        print(
            generate_prompt.format(
                question=question,
                context=relevant_context,
            )
        )
        generator = generate_prompt | llm | StrOutputParser()

        result = generator.invoke(
            input={
                "question": question,
                "context": relevant_context,
            }
        )
        print("___" * 100)
        print(result)
        return {"generation": result}

    def buildGRaph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("web_search", self.web_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_node("grade_relevance", self.gradeContext)

        workflow.set_conditional_entry_point(
            self.routingPrompt,
            # JSON output from routing mapped to function call
            {
                "web_search": "web_search",
                "vector_store": "retrieve",
            },
        )

        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_relevance")
        workflow.add_conditional_edges(
            "grade_relevance",
            self.decideToGenerate,
            {"web_search": "web_search", "generate": "generate"},
        )
        workflow.add_edge("generate", END)
        # workflow.add_edge("retrieve", END)
        runnable = workflow.compile()
        return runnable
