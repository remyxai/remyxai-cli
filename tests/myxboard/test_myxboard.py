import logging
import traceback
from remyxai.client.myxboard import MyxBoard
from remyxai.client.remyx_client import RemyxAPI
from remyxai.api.evaluations import EvaluationTask

# Set up logging to include more detailed information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

def test_myxboard_evaluation():
    try:
        # Step 1: Initialize MyxBoard with a list of model identifiers
        model_ids = ["Phi-3-mini-4k-instruct", "Qwen2-1.5B"]
        myx_board_name = "test_myxboard_new_3"
        myx_board = MyxBoard(model_repo_ids=model_ids, name=myx_board_name)
        logging.info(f"MyxBoard {myx_board_name} initialized with models: {model_ids}")

        # Step 2: Initialize the evaluation API client
        remyx_api = RemyxAPI()
        logging.info("Initialized RemyxAPI client")

        # Step 3: Define the tasks you want to evaluate
        tasks = [EvaluationTask.MYXMATCH]
        prompt = "What are 2 characteristics of a good employee?"

        # Step 4: Evaluate the entire MyxBoard on all specified tasks
        remyx_api.evaluate(myx_board, tasks, prompt=prompt)
        logging.info(f"Evaluation started for MyxBoard {myx_board_name} with tasks: {tasks}")

        # Step 5: Fetch the updated results from the MyxBoard
        results = myx_board.get_results()
        logging.info(f"Results fetched for MyxBoard {myx_board_name}: {results}")

        # Step 6: Print the results
        print("Evaluation Results:")
        print(results)

    except Exception as e:
        # Log the full traceback in case of any exception
        logging.error(f"An error occurred during the MyxBoard evaluation test: {str(e)}")
        logging.error(traceback.format_exc())  # Log the full traceback for detailed debugging

# Run the test function
if __name__ == "__main__":
    test_myxboard_evaluation()

