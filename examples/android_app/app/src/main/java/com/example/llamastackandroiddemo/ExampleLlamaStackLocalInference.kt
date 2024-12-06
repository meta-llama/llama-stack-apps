package com.example.llamastackandroiddemo

import android.util.Log
import com.llama.llamastack.client.LlamaStackClientClient
import com.llama.llamastack.client.local.LlamaStackClientLocalClient
import com.llama.llamastack.core.JsonNumber
import com.llama.llamastack.models.CompletionMessage
import com.llama.llamastack.models.InferenceChatCompletionParams
import com.llama.llamastack.models.SystemMessage
import com.llama.llamastack.models.UserMessage
import java.util.concurrent.CompletableFuture

class ExampleLlamaStackLocalInference(
    val modelPath: String,
    val tokenizerPath: String,
    val temperature: Float
) {

    private var client: LlamaStackClientClient? = null
    private var response: String? = null
    private var tps: Float = 0.0f

    fun getResponse(): String? {
        return response
    }

    fun getTps(): Float {
        return tps
    }

    init {
        val thread = Thread {
            try {
                Log.d("llama_stack","ExampleLlamaStackLocalInference init is called")
                client = LlamaStackClientLocalClient
                    .builder()
                    .modelPath(modelPath)
                    .tokenizerPath(tokenizerPath)
                    .temperature(temperature)
                    .build()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        thread.start();
    }

    fun updateModel(modelPath: String, tokenizerPath: String, temperature: Float) {
        val thread = Thread {
            try {
                AppLogging.getInstance().log("Updating local model to $modelPath")
                // For now, we just create a new client, but eventually, we don't want this, because dealloc of resetNative()
                // is not called here.
                // Need to de-alloc and call resetNative() from ET side
                client =  LlamaStackClientLocalClient
                    .builder()
                    .modelPath(modelPath)
                    .tokenizerPath(tokenizerPath)
                    .temperature(temperature)
                    .build()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        thread.start();
    }

    fun inferenceStart(modelName: String, conversationHistory: ArrayList<Message>, systemPrompt:String): String {
        val future = CompletableFuture<String>()
        val thread = Thread {
            try {
                val response = inferenceCall(modelName, conversationHistory, systemPrompt).toString();
                future.complete(response)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        thread.start();
        return future.get();
    }

    private fun inferenceCall(modelName: String, conversationHistory: ArrayList<Message>, systemPrompt: String): String? {
        // Multi prompt/ chat history use case
        if (client == null) {
            AppLogging.getInstance().log("client is null for local inference");
            return "[ERROR] client is null for local inference"
        }
        AppLogging.getInstance().log("local inference with prompt=${conversationHistory.last().text}")
        val sequenceLength = ModelUtils.getSequenceLengthForConversationHistory(conversationHistory, systemPrompt)
        val result = client!!.inference().chatCompletion(
            InferenceChatCompletionParams.builder()
                .modelId(modelName)
                .putAdditionalQueryParam("seq_len", sequenceLength.toString())
                .messages(
                    constructLSMessagesFromConversationHistoryAndSystemPrompt(conversationHistory, systemPrompt)
                )
                .build()
        )
        response = result.asChatCompletionResponse().completionMessage().content().string();
        tps =
            (result.asChatCompletionResponse()._additionalProperties()["tps"] as JsonNumber).value as Float
        return response;
    }

    private fun constructLSMessagesFromConversationHistoryAndSystemPrompt(
        conversationHistory: ArrayList<Message>,
        systemPrompt: String
    ):List<InferenceChatCompletionParams.Message> {
        val messageList = ArrayList<InferenceChatCompletionParams.Message>();
        // System prompt
        messageList.add(
            InferenceChatCompletionParams.Message.ofSystemMessage(
                SystemMessage.builder()
                    .content(SystemMessage.Content.ofString(systemPrompt))
                    .role(SystemMessage.Role.SYSTEM)
                    .build()
            )
        )
        // User and assistant messages
        for (chat in conversationHistory) {
            var inferenceMessage: InferenceChatCompletionParams.Message
            if (chat.isSent) {
                // User message
                inferenceMessage = InferenceChatCompletionParams.Message.ofUserMessage(
                    UserMessage.builder()
                        .content(UserMessage.Content.ofString(chat.text))
                        .role(UserMessage.Role.USER)
                        .build()
                )
            } else {
                // Assistant message (aka previous prompt response)
                inferenceMessage = InferenceChatCompletionParams.Message.ofCompletionMessage(
                    CompletionMessage.builder()
                        .content(CompletionMessage.Content.ofString(chat.text))
                        .stopReason(CompletionMessage.StopReason.END_OF_MESSAGE)
                        .toolCalls(emptyList())
                        .role(CompletionMessage.Role.ASSISTANT)
                        .build()
                )
            }
            messageList.add(inferenceMessage)
        }
        AppLogging.getInstance().log("conversation history length "  + messageList.size)
        return messageList
    }

}