# Eval Examples

The folders in this script demostrates examples to use Llama Stack Evals API to run evaluation.


## Benchmark Evals

### MMLU
```
python -m examples.eval.mmlu localhost 5000 [--dataset_path <optional preprocessed dataset path>] [--strict True/False]
```

Example outputs:
```
EvalEvaluateResponse(generations=[{'generated_answer': 'The correct procedure when a blood clot is causing issues with a catheter is to inform medical staff so they can take the necessary steps to address the issue, which may include administering anticoagulants or flushing the catheter.\n\nAnswer: B)'}, {'generated_answer': 'Answer: D'}, {'generated_answer': 'Answer: C'}], scores={'meta-reference::answer_parsing_multiple_choice': Scores(aggregated_results={'accuracy': 1.0, 'num_correct': 3.0, 'num_total': 3.0}, score_rows=[{'score': 1.0}, {'score': 1.0}, {'score': 1.0}])})
```
