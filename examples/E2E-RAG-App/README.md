## E2E-RAG-App

This is an E2E RAG App that can be pointed to any folder to do RAG over a collection of mixed file formats and do retrieval using the `Llama-3.2-3B-Instruct` Model

Details:
TODO:
1. Save memory_bank to local, and load it to local
2. Make the chat inference multi-turn
3. Front-end + docker

To run the `ingestion_script.py` script, please make sure there is a /DATA and /OUTPUT folder at its relative root. It will ingest ALL documents in /DATA and output BOTH markdown and JSON dump in /OUTPUT folder



```
~/work/llama-stack-apps/examples/E2E-RAG-App (rag-app)]$ python rag_main.py localhost 5000 ./example_data/
Inserted 1 documents into bank: rag_agent_docs
Created bank: rag_agent_docs
Found 2 models [ModelDefWithProvider(identifier='Llama3.2-11B-Vision-Instruct', llama_model='Llama3.2-11B-Vision-Instruct', metadata={}, provider_id='meta-reference', type='model'), ModelDefWithProvider(identifier='Llama-Guard-3-1B', llama_model='Llama-Guard-3-1B', metadata={}, provider_id='meta1', type='model')]
Use model: Llama3.2-11B-Vision-Instruct
  0%|                                                                                           | 0/1 [00:00<?, ?it/s]Generating response for: What methods are best for finetuning llama models?
100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [00:27<00:00, 27.59s/it]
Based on the provided context, the best methods for fine-tuning Llama models are not explicitly stated, but some general guidelines can be inferred. Here are some possible methods:

1. **Data augmentation**: Fine-tuning a Llama model typically requires a large amount of high-quality data. You can try data augmentation techniques, such as text synthesis, paraphrasing, or back-translation, to increase the size of your dataset.
2. **Task-specific fine-tuning**: Instruction-tuned models, like Llama 3, are designed for specific tasks, such as conversational AI. Fine-tuning these models for your specific task can improve their performance.
3. **Gradual fine-tuning**: Gradually fine-tuning a Llama model can help to maintain its original properties while adapting it to your specific task. This approach involves fine-tuning the model in stages, with each stage focusing on a specific aspect of the task.
4. **Hybrid fine-tuning**: You can combine multiple fine-tuning methods to achieve better results. For example, you can use a combination of task-specific fine-tuning and data augmentation to fine-tune your Llama model.
5. **Evaluation metrics**: Regularly evaluate your fine-tuned model using metrics such as accuracy, F1 score, or BLEU score. This will help you identify areas where the model needs improvement.
6. **Hyperparameter tuning**: Hyperparameter tuning is essential for fine-tuning a Llama model. Experiment with different hyperparameters, such as learning rate, batch size, and number of epochs, to find the optimal combination for your specific task.
7. **Red teaming exercises**: As mentioned in the provided context, red teaming exercises are a technique used to identify potential weaknesses in a Llama model. Conducting red teaming exercises can help you identify areas where the model may not be performing optimally.
8. **Model refusals**: Model refusals are an important aspect of Llama model fine-tuning. The context mentions that the developers have improved the model refusals to ensure that Llama 3 is significantly less likely to falsely refuse to answer prompts.
9. **Use of safety tools**: The context also mentions the use of safety tools, such as Meta Llama Guard 2 and Code Shield, to reduce residual risks in LLM systems. You can explore the use of these safety tools to fine-tune your Llama model.

Some code-based methods for fine-tuning Llama models are available in the `llama-recipes` repository on GitHub. You can also consult the official Llama documentation for more information on fine-tuning and using Llama models.

In terms of specific libraries or tools, the following can be useful for fine-tuning Llama models:

* Hugging Face Transformers library: This library provides a wide range of pre-trained models, including Llama models, and allows for easy fine-tuning.
* PyTorch: This library provides a dynamic computation graph and is widely used for deep learning tasks, including fine-tuning Llama models.
* TensorFlow: This library provides a static computation graph and is widely used for deep learning tasks, including fine-tuning Llama models.

Keep in mind that fine-tuning a Llama model requires a significant amount of computational resources and expertise in deep learning. It's essential to carefully evaluate the model's performance and adjust the fine-tuning parameters accordingly.
```
