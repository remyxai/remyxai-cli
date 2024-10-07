import logging
import traceback
from remyxai.client.myxboard import MyxBoard
from remyxai.client.remyx_client import RemyxAPI
from remyxai.api.evaluations import EvaluationTask

# Set up logging to include more detailed information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

def test_myxboard_evaluation():
    try:
        model_ids = ["Phi-3-mini-4k-instruct", "Qwen2-1.5B"]
        myx_board_name = "test_myxboard"
        myx_board = MyxBoard(model_repo_ids=model_ids, name=myx_board_name)
        logging.info(f"MyxBoard {myx_board_name} initialized with models: {model_ids}")

        remyx_api = RemyxAPI()
        logging.info("Initialized RemyxAPI client")

        tasks = [EvaluationTask.MYXMATCH]
        prompt = "What are 2 characteristics of a good coder?"

        remyx_api.evaluate(myx_board, tasks, prompt=prompt)
        logging.info(f"Evaluation started for MyxBoard {myx_board_name} with tasks: {tasks}")

        results = myx_board.get_results()
        logging.info(f"Results fetched for MyxBoard {myx_board_name}: {results}")

        print("Evaluation Results:")
        print(results)

    except Exception as e:
        logging.error(f"An error occurred during the MyxBoard evaluation test: {str(e)}")
        logging.error(traceback.format_exc())

def test_myxboard_evaluationi_hf(collection_name):
    try:
        myx_board =  MyxBoard(hf_collection_name=collection_name)
        logging.info(f"MyxBoard {myx_board_name} initialized from collection: {collection_name}")

        remyx_api = RemyxAPI()
        logging.info("Initialized RemyxAPI client")

        tasks = [EvaluationTask.MYXMATCH]
        prompt = "What are 2 characteristics of a good coder?"

        remyx_api.evaluate(myx_board, tasks, prompt=prompt)
        logging.info(f"Evaluation started for MyxBoard {myx_board_name} with tasks: {tasks}")

        results = myx_board.get_results()
        logging.info(f"Results fetched for MyxBoard {myx_board_name}: {results}")

        print("Evaluation Results:")
        print(results)

        # Optional: upload results to collection
        myx_board.push_to_hf()


    except Exception as e:
        logging.error(f"An error occurred during the MyxBoard evaluation test: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    test_myxboard_evaluation()

