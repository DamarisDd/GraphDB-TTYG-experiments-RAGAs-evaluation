# ------------------------------------------------------------------------------
# RAGAs evaluation script for LLM responses on queries regarding a BPMN process (GraphDB's TTYG SPARQL query method)
#
# IMPORTANT:
# - If you encounter an error related to the number of tokens being exceeded (depends on the GPT model used), simply comment out some lines in the
#   'questions', 'answers' and 'ground_truths' lists and run the script in batches.
# - Each time you run a batch, be sure to change the exported file name, so your previous results do not get overwritten.
# ------------------------------------------------------------------------------

import os
import json
import numpy as np
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ResponseRelevancy,
    SemanticSimilarity,
    Faithfulness
)
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate

# 1. Load ontology as context
loader = TextLoader("ontology.txt")
documents = loader.load()
# Extract and clean the full text content
full_context = " ".join([doc.page_content.strip() for doc in documents])  # Remove leading/trailing spaces
full_context = full_context.replace("\n", " ")  # Replace newlines with spaces
full_context = " ".join(full_context.split())  # Normalize multiple spaces

# 2. Define data for evaluation
questions = [
    # Level 2
    'Is "Return undelivered product" a manual task?',
    'What is the task type of "Settle with merchant"?',
    'Is "Ship replacement" marked for compensation?',
    'Does "Cancel invoice" act as a compensation task?',
    'Is there any task that has a defined cost?',
    'Does "Issue invoice" have a defined execution time?',
    'Is the "Customer still interested?" event conditional?'
    #########
    # Level 3
    'Who is the participant that cancels the order?',
    'How long does it take to transport the parcel to the client?',
    'In case the filed claim is accepted, how long does it take to ship the replaced product?',
    'Who is responsible for processing returns?',
    'Can you list all tasks that are supported by software?',
    'Which tasks rely on sending messages?',
    'Is there a point in the process where the customer’s interest determines the next step? If so, which one is it and what are the next steps?',
    'Which user task has the highest execution cost and what is its value? Who is responsible for that particular task and what is the hourly wage for that responsible individual?',
    'What task is performed by the courier right before the parcel is delivered to the customer?',
    'Right after issuing an invoice, the merchant cancels the invoice. Is this true? If not, correct the statement.',
    'There are no tasks that occur after the order is marked as undelivered.',
    'After receiving the parcel, the client either uses the product or settles with the merchant.'
    #########
    # Level 4
    'List all user tasks from the subprocess that investigates lost orders, that are under the responsibility of the Customer service agent and have an execution time of less than 5 minutes. For each, specify the subsequent task. A good output should be con-cise, of less than 100 words.',
    'Some processes may contain gateways with no outgoing sequence flows. List these gateways and for each, indicate whether this is likely a modeling error or an intentional dead-end. A good output should be concise, fitting in a non-bulleted style paragraph of less than 200 words.'
]

ground_truths = [
    # Level 2
    'Yes, "Return undelivered product" is a manual task.',
    'The task type of "Settle with merchant" is User.',
    'No, "Ship replacement" is not marked for compensation.',
    'Yes, "Cancel invoice" acts as a compensation task.',
    'Yes, there are multiple tasks that each have a defined cost.',
    'Yes, "Issue invoice" has a defined execution time.',
    'No, the "Customer still interested?" is not a conditional event.'
    #########
    # Level 3
    'The participant that cancels the order is the merchant.',
    'It takes 2 days to transport the parcel to the client.',
    'In case the filed claim is accepted, it takes 30 minutes to ship the replaced product.',
    'The responsible for processing returns is the returns processor.',
    '''The tasks that are supported by software are:
    Log complaint
    Issue invoice
    Contact customer
    Settle with merchant
    Cancel invoice
    Cancel order
    Process return
    Check tracking information
    File claim with courier
    Cancel order & notify customer''',
    'The tasks that rely on sending messages are "Notify merchant" and "Notify customer of new estimated time of arrival".',
    'Yes, there is a point in the process where the customer’s interest determines the next step. This is the "Customer still interested?" exclusive gateway and the next steps are "Ship replacement" if the customer is interested or "Cancel order & notify customer" if the customer is not interested.',
    'The user task with the highest execution cost is "File claim with courier" with a value of 5. The responsible individual for this particular task is the "Claims specialist" with an hourly wage of 18.',
    'The task performed by the courier right before the parcel is delivered to the customer is "Contact customer".',
    'No, this is not true. Right after issuing an invoice, the merchant packs the order.',
    'This statement is not true. There are four tasks for compensation that occur after the order is marked as undelivered: "Return undelivered product", "Return to warehouse", "Cancel invoice" and "Cancel order".',
    'This statement is partially correct. After receiving the parcel, the client uses the product and in parallel, the courier settles with the merchant.'
    #########
    # Level 4
    'The user tasks from the subprocess that investigates lost orders, that is under the responsibility of the Customer service agent and has an execution time of less than 5 minutes is "Check tracking information". The subsequent task is either "Notify customer of new estimated time of arrival" or "File claim with courier".',
    'The processes do not contain gateways with no outgoing sequence flows.'
]

answers = [
    # Level 2
    'Yes, "Return undelivered product" is a manual task.',
    'The task type of "Settle with merchant" is User.',
    'No, "Ship replacement" is not marked for compensation',
    'Yes, "Cancel invoice" acts as a compensation task.',
    'Yes, there is at least one task in the process that has a defined cost.',
    'Yes, "Issue invoice" has a defined execution time of 1 minute.',
    'No, the "Customer still interested?" event is not marked as conditional in the process model.'
    #########
    # Level 3
    'The participant responsible for cancelling the order is the "Invoicing clerk".',
    'It takes 2 days to transport the parcel to the client.',
    'If the filed claim is accepted, it takes 30 minutes to ship the replaced product.',
    'The person responsible for processing returns is the "Returns processor."',
    '''The tasks that are supported by software (task type "Service") are:
    Cancel invoice
    Issue invoice''',
    '''The tasks that rely on sending messages are:
    Notify merchant
    Notify customer of new estimated time of arrival''',
    '''Yes, there is a point in the process where the customer’s interest determines the next step. This occurs at the gateway labeled "Customer still interested?".
    The possible next steps from this gateway are:
    "Cancel order & notify customer"
    "Ship replacement"
    So, depending on whether the customer is still interested, the process will either proceed to cancel the order and notify the customer, or to ship a replacement.''',
    'The user task with the highest execution cost is "File claim with courier" with a cost of 5. The responsible individual for this task is the "Claims specialist," whose hourly wage is 18.',
    'The task performed by the courier right before delivering the parcel to the customer is reaching the decision point "Customer found?" (an exclusive gateway). This gateway determines whether the courier can proceed to deliver the parcel to the customer.',
    'The statement is not true. Right after issuing an invoice, the next step is "Pack order," not "Cancel invoice." The correct sequence is: "Issue invoice" → "Pack order."',
    '''That statement is not correct. After the order is marked as undelivered (at the "Order undelivered" end event), there are still tasks that occur in the process model. Specifi-cally, after "Order undelivered," the following tasks are executed:
    Return undelivered product
    Return to warehouse
    Cancel invoice
    These tasks ensure the undelivered product is handled and the invoice is canceled, even after the order is marked as undelivered.''',
    '''Your statement is almost correct, but let's clarify the process flow based on the BPMN model:
    After the client receives the parcel ("Receive parcel - Cash on delivery"), the process continues to a non-exclusive gateway ("Non-exclusive Gateway (BPMN)-31871"). From this gateway, two parallel paths are possible:
    Use product: The client can proceed to use the product.
    Settle with merchant: The process can also proceed to settling with the merchant.
    These two actions are not mutually exclusive; both can occur in parallel or independently, according to the process model. Summary: After receiving the parcel, the client may use the product and/or settle with the merchant. Both actions are possible and can happen in parallel.'''
    #########
    # Level 4
    'The only user task in the "Lost order investigation subprocess" under the responsibility of the Customer service agent with an execution time of less than 5 minutes is "Check tracking information". Its subsequent task is "Package located?".',
    'All gateways in the provided models have at least one outgoing sequence flow; none are found without outgoing flows. Therefore, there are no gateways that represent either a modeling error or an intentional dead-end in this context. If such gateways existed, they would typically indicate a modeling error, as gateways are meant to direct process flow, and having no outgoing sequence flows would leave the process stranded. However, since none are present, this issue does not arise in your models.'
]

# Use the cleaned full context
contexts = [[full_context] for _ in questions]  # Every question gets the same cleaned file as context

# Dataset construction
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths,
    "reference": ground_truths
}

dataset = Dataset.from_dict(data)

# 3. Setup OpenAI API and LLM
load_dotenv(find_dotenv())  # Loads .env file if present

llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.2, top_p=1)
evaluator_llm = LangchainLLMWrapper(llm)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# RAGAs metrics setup
response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=embeddings)
factual_correctness = FactualCorrectness(llm=evaluator_llm, mode="precision", atomicity="low", coverage="high")
semantic_similarity = SemanticSimilarity(embeddings=LangchainEmbeddingsWrapper(embeddings))
faithfulness = Faithfulness(llm=evaluator_llm)

# 4. Evaluate responses
try:
    print("Evaluating answers...\n")
    result = evaluate(
        dataset=dataset,
        metrics=[
            response_relevancy,
            factual_correctness,
            semantic_similarity,
            faithfulness
        ]
    )
    df = result.to_pandas().round(2)

    # Construct evaluation records (exclude context for brevity)
    evaluations = []
    for i in range(len(df)):
        record = {
            "question": questions[i],
            "answer": answers[i],
            "ground_truth": ground_truths[i],
            "answer_relevancy": df["answer_relevancy"].iloc[i],
            "factual_correctness": df["factual_correctness"].iloc[i],
            "semantic_similarity": df["semantic_similarity"].iloc[i],
            "faithfulness": df["faithfulness"].iloc[i],
        }
        evaluations.append(record)

    final_output = {"evaluations": evaluations}

    # 5. Save output in a specific subfolder
    output_folder = os.path.join(os.path.dirname(__file__), "RAGAs_results_SPARQL_method")
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    output_path = os.path.join(output_folder, "results_SPARQL.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"\nJSON export: successful. File saved at: {output_path}")

except Exception as e:
    print(f"Error during evaluation: {e}")