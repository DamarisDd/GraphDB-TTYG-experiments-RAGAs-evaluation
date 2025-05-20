# ------------------------------------------------------------------------------
# RAGAs evaluation script for LLM responses on queries regarding a BPMN process (GraphDB's TTYG ChatGPT Retrieval Connector query method)
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

# 1. Load context chunks (Weaviate export) as input
loader = TextLoader("retrieved_context_for_LLM.txt")
documents = loader.load()

# Clean and normalize full context
full_context = " ".join([doc.page_content.strip() for doc in documents])  # Remove leading/trailing spaces
full_context = full_context.replace("\n", " ")  # Replace newlines with spaces
full_context = " ".join(full_context.split())  # Collapse multiple spaces

# 2. Define evaluation data
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
    'Some processes may contain gateways with no outgoing sequence flows. List these gateways and for each, indicate whether this is likely a modeling error or an intentional dead-end. A good output should be concise, fitting in a non-bulleted style paragraph of less than 200 words.',
    'For each participant, return: (1) the number of tasks they perform; (2) the types of events belonging to each participant’s process flow; (3) the name of any decision points (exclusive gateways) within each participant’s process scope. A good output should: be concise, of less than 150 words, and use a numbered list format.'
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
    'The processes do not contain gateways with no outgoing sequence flows.',
    '''1. Customer:
(1) The number of performed tasks: 10
(2) The types of events belonging to the Customer's process flow: Start, End, Intermediate, Boundary
(3) The name of any decision points (exclusive gateways) within each participant’s process scope: none.
2. Merchant:
(1) The number of performed tasks: 12
(2) The types of events belonging to the Customer's process flow: Start, End, Intermediate, Boundary
(3) The name of any decision points (exclusive gateways) within each participant’s process scope: none.
3. Courier:
(1) The number of performed tasks: 6
(2) The types of events belonging to the Customer's process flow: End, Boundary
(3) The name of any decision points (exclusive gateways) within each participant’s process scope: Customer found?.'''
]

answers = [
    # Level 2
    'Yes, "Return undelivered product" is a manual task.',
    'The task type of "Settle with merchant" is User.',
    'No, the task "Ship replacement" is not marked for compensation.',
    'Yes, "Cancel invoice" acts as a compensation task.',
    '''Yes, there are several tasks that have a defined cost. For example:
•	"Cancel invoice" has a cost of 0.1.
•	"Ship replacement" has a cost of 6.
•	"Return undelivered product" has a cost of 8.
•	"Return product to courier" has a cost of 3.5.
•	"Deliver parcel to customer" has a cost of 4.
''',
    'Yes, the task "Issue invoice" has a defined execution time of 00:000:00:01:00 (1 minute).',
    'The "Customer still interested?" event is not a conditional event. It is modeled as an Exclusive Gateway of type "Data-based (without marker)", not as a conditional event. So, while it represents a decision point in the process, it is not technically a BPMN conditional event.'
    #########
    # Level 3
    'The participant that cancels the order is the Merchant. This is indicated by the task "Cancel order," which is executed by the participant Merchant and is under the responsibility of the Invoicing clerk.',
    'The task "Transport parcel" has a defined execution time of 00:002:00:00:00, which means it takes 2 days to transport the parcel to the client.',
    'If the filed claim is accepted, the task to "Ship replacement" has a defined execution time of 00:000:00:30:00, which means it takes 30 minutes to ship the replaced product.',
    'The person responsible for processing returns is the Returns Processor. This role is part of the organizational unit "Logistics" within the Merchant Organization Structure. The Returns Processor is specifically mentioned as being under the responsibility for the "Process return" task.',
    '''Based on the available data, the following tasks are supported by software (i.e., are of type "Service" or otherwise automated):
    1.	Issue invoice – Task type: Service
    2.	Cancel invoice – Task type: Service
    These tasks are explicitly marked as "Service" tasks, which indicates they are supported by software. Other tasks listed are manual or user tasks, which require human intervention.''',
    '''The following tasks rely on sending messages:
    1.	Notify merchant
    Task type: Send
    Executed by: Customer
    2.	Notify customer of new estimated time of arrival
    Task type: Send
    Executed by: Customer service agent
    These tasks are explicitly of the "Send" type, indicating they involve sending messag-es as part of their execution.''',
    'Yes, there is a point in the process where the customer’s interest determines the next step. This is at the step "Customer still interested?" If the customer is still interested, the process leads to "Ship replacement." If not, it leads to "Cancel order & notify customer."',
    'The user task with the highest execution cost is "File claim with courier," which has a cost of 5. The person responsible for this task is the Claims specialist, whose hourly wage is 18.',
    'The task performed by the courier right before "Deliver parcel to customer" is the decision point "Customer found?". This is an exclusive gateway where the courier determines if the customer is available to receive the parcel. If the customer is found, the process proceeds to "Deliver parcel to customer."',
    'The statement is not true. After the order is marked as undelivered, there is a subsequent task: "Return undelivered product." This task is executed by the courier. Therefore, tasks do occur after the order is marked as undelivered.',
    'You are correct. There are no tasks (BPMN) that occur after the order is marked as "Order undelivered." The process element that follows is an End Event ("Order undelivered"), which signifies the termination of that process path.',
    'The statement is not accurate. After receiving the parcel ("Receive parcel - Cash on delivery"), the next step is a non-exclusive gateway, which means multiple paths may follow. However, the available context does not confirm that the client either uses the product or settles with the merchant as the immediate next steps. The process may involve other actions or decisions after receiving the parcel.'
    #########
    # Level 4
    'In the lost order investigation subprocess, the Customer service agent performs two user tasks with execution times under five minutes: "Check tracking information" (3 minutes), which is followed by "Package located?," and "Notify customer of new estimated time of arrival" (2 minutes), which is followed by "Investigation closed."',
    'The gateways "Package located?", "Customer still interested?", "Claim accepted?", and "Exclusive Gateway (BPMN)-30689" are listed as having no outgoing sequence flows. In standard BPMN modeling, gateways without outgoing flows typically indicate a modeling error, as gateways are intended to direct process flow based on conditions or events. Unless these gateways are deliberately used as placeholders for incomplete models or to mark intentional process termination points (which is rare and not best practice), their lack of outgoing flows is most likely unintentional and should be reviewed for correction. There is no evidence in the retrieved data suggesting these are intentional dead-ends, so they are best interpreted as modeling oversights.',
    '''1. Customer:
Tasks: At least 1 (e.g., “Search product at various merchants”).
Event types: Conditional event (e.g., “Product problems”), Compensation event.
Exclusive gateways: None named within scope.
2. Courier:
Tasks: At least 3 (“Deliver return”, “Return undelivered product”, “Contact customer”).
Event types: Compensation event, Manual task.
Exclusive gateways: “Customer found?” (Data-based).
3. Merchant:
Tasks: At least 1 (“Process return”).
Event types: Compensation event.
Exclusive gateways: None named within scope.
Note: Some exclusive gateways (e.g., “Customer found?”) are data-based decision points. Other gateways and events may exist, but only those explicitly linked to participants are listed.'''
]

# Use the cleaned full context
contexts = [[full_context] for _ in questions]  # Every question gets the same cleaned file as context

# Construct the dataset
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths,
    "reference": ground_truths
}
dataset = Dataset.from_dict(data)

# 3. LLM and metrics setup
load_dotenv(find_dotenv())  # Load .env if present

llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.2, top_p=1)
evaluator_llm = LangchainLLMWrapper(llm)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=embeddings)
factual_correctness = FactualCorrectness(llm=evaluator_llm, mode="precision", atomicity="low", coverage="high")
semantic_similarity = SemanticSimilarity(embeddings=LangchainEmbeddingsWrapper(embeddings))
faithfulness = Faithfulness(llm=evaluator_llm)

# 4. Evaluate and save results to a subfolder
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

    # Build evaluation records
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

    # Output folder
    output_folder = os.path.join(os.path.dirname(__file__), "RAGAs_results_ChatGPTRetrievalConnector_method")
    os.makedirs(output_folder, exist_ok=True)  # Create if not exists

    output_path = os.path.join(output_folder, "results_ChatGPTRetrievalConnector.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"\nJSON export: successful. File saved at: {output_path}")

except Exception as e:
    print(f"Error during evaluation: {e}")
