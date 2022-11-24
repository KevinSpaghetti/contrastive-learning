import evaluate
from evaluate.utils import launch_gradio_widget

module = evaluate.load("KevinSpaghetti/accuracyk")
launch_gradio_widget(module)