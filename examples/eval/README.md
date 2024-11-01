# Eval Examples

The folders in this script demostrates examples to use Llama Stack Evals API to run evaluation.


## Benchmark Evals

### MMLU
```
python -m examples.eval.mmlu localhost 5000 [--dataset_path <optional preprocessed dataset path>] [--strict True/False]
```

Example outputs:
```
Saved dataset at /data/users/xiyan/llama-stack-apps/examples/eval/Llama-3.2-1B-Instruct-evals__mmlu__details-loose.csv!
Eval result saved at /data/users/xiyan/llama-stack-apps/examples/eval/eval-result.json!
```

Example results
```
$ cat eval-result.json

{
    "generations": [
        {
            "generated_answer": "The best course of action in this situation would be to inform medical staff about the issue as they can assess the situation, determine the best course of action, and potentially take steps to prevent further clotting or replace the catheter if necessary. \n\nAnswer: B."
        },
        {
            "generated_answer": "Answer: D"
        },
        {
            "generated_answer": "Global inequality can lead to various social, economic, and political instability, and some of the ways it might pose a threat to global security involve:\n\n- As options B and C describe, disenfranchised populations who feel they have been left behind can become increasingly dissatisfied with their situation. This dissatisfaction can eventually lead to discontent, protests, and potential violent uprisings as seen in the Arab Spring or in anti-austerity movements. \n\nOption B contradicts the threat that discontent populations pose, while option D is too dismissive of the implications of global inequality.\n\nGiven the potential of global inequality to fuel unrest and radicalization among populations left behind by globalisation, the best answer would reflect this dynamic and potential consequence. \n\nAnswer: C"
        }
    ],
    "scores": {
        "meta-reference::answer_parsing_multiple_choice": {
            "aggregated_results": {
                "accuracy": 1.0,
                "num_correct": 3.0,
                "num_total": 3.0
            },
            "score_rows": [
                {
                    "score": 1.0
                },
                {
                    "score": 1.0
                },
                {
                    "score": 1.0
                }
            ]
        }
    }
}
```
