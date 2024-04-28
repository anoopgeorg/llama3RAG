from flask import Flask, request
from dotenv import load_dotenv, dotenv_values

# from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.globals import set_debug
import os

from src.GraphState import GraphState
from src.prompts import Prompter
from src.vault import Vault
from src.entities.data_entity import UpdateVault


application = Flask(__name__)

# set_debug(True)
load_dotenv()
print(os.getenv("TAVILY_API_KEY"))
# cached_llm = Ollama(model="llama3")
cached_llm = ChatOllama(model="llama3", temperature=0.3)

prompter = Prompter(llm=cached_llm)
vault = Vault()


def start_app():
    application.run(host="0.0.0.0", port=8080, debug=True)


@application.route("/")
def landingPage():
    return "Hello"


@application.route("/askAlfy", methods=["POST"])
def askAlfy():
    json_content = request.json
    query = json_content.get("query")
    response = prompter.simplePrompt(query=query).get("text")
    return response


@application.route("/askAlfyKnowledge", methods=["POST"])
def askAlfyKnowledge():
    json_content = request.json
    query = json_content.get("query")
    # retriever = vault.getRetriever()
    response = prompter.documentPrompt(query=query)
    return response


@application.route("/router", methods=["POST"])
def router():
    json_content = request.json
    query = json_content.get("query")
    response = prompter.routingPrompt(query=query)
    return response


@application.route("/updateVault", methods=["POST"])
def updateVault():
    print("Entering vault update******************")
    form_content = request.form
    update_entity = UpdateVault(
        type=form_content.get("type"), file=request.files["file"]
    )
    response = vault.updateVault(obj=update_entity)
    return response


@application.route("/traverse", methods=["POST"])
def retriever():
    json_content = request.json
    query = json_content.get("query")

    graph_traverser = prompter.buildGRaph()
    result = graph_traverser.invoke({"query": query})
    print(result)
    return result["route"]


if __name__ == "__main__":
    start_app()
