package com.example.llamastackandroiddemo

import android.content.Context
import android.util.Log
import com.llama.llamastack.client.LlamaStackClientClient
import com.llama.llamastack.client.local.LlamaStackClientLocalClient
import com.llama.llamastack.core.JsonNumber
import com.llama.llamastack.models.CompletionMessage
import com.llama.llamastack.models.InferenceChatCompletionParams
import com.llama.llamastack.models.SystemMessage
import com.llama.llamastack.models.UserMessage
import kotlinx.datetime.Clock
import java.time.Instant
import java.time.ZoneId
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
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

    fun inferenceStart(modelName: String, conversationHistory: ArrayList<Message>, systemPrompt:String, ctx: Context): String {
        val future = CompletableFuture<String>()
        //Get the current time in ISO format and pass it to the model in system prompt as a reference. This is useful for any scheduling and vague timing reference from user prompt.
        val zdt = ZonedDateTime.ofInstant(Instant.parse(Clock.System.now().toString()), ZoneId.systemDefault())
        val formattedZdt = zdt.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"))
        val availableFunctions = AvailableFunctions.getInstance()
        val functionDefinitions = availableFunctions.values()
        var sysPrompt = systemPrompt
        if (sysPrompt == "") {
            sysPrompt = """
                            Today Date: $formattedZdt
                    
                            Tool Instructions:
                            - When user is asking a question that requires your reasoning, do not use a function call or generate functions.
                            - Only function call if user's intention matches the function that you have access to.
                            - When looking for real time information use relevant functions if available.
                            - Ignore previous conversation history if you are making a tool call.

                                           
                            You have access to the following functions:
                            {$functionDefinitions}
                                      
                            If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n
                            You SHOULD NOT include any other text in the response.
                    
                            Reminder:                          
                            - Function calls MUST follow the specified format
                            - Required parameters MUST be specified
                            - Only call one function at a time
                            - Put the entire function call reply on one line
                            - When returning a function call, don't add anything else to your response
                            - When scheduling the events, make sure you set the date and time right. Use step by step reasoning for date such as next Tuesday
                         """
        }
        val thread = Thread {
            try {
                val response = inferenceCall(modelName, conversationHistory, sysPrompt, ctx);
                future.complete(response)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        thread.start();
        return future.get();
    }

    private fun inferenceCall(modelName: String, conversationHistory: ArrayList<Message>, systemPrompt: String, ctx: Context): String? {
        // Multi prompt/ chat history use case
        if (client == null) {
            AppLogging.getInstance().log("client is null for local inference");
            return "[ERROR] client is null for local inference"
        }
        AppLogging.getInstance().log("local inference with prompt=${conversationHistory.last().text}")
        val sequenceLength = ModelUtils.getSequenceLengthForConversationHistory(conversationHistory, systemPrompt)
        var response = ""
        val result = client!!.inference().chatCompletion(
            InferenceChatCompletionParams.builder()
                .modelId(modelName)
                .putAdditionalQueryParam("seq_len", sequenceLength.toString())
                .messages(
                    constructLSMessagesFromConversationHistoryAndSystemPrompt(conversationHistory, systemPrompt)
                )
                .build()
        )
        response = result.asChatCompletionResponse().completionMessage().content().string().toString();
        tps =
            (result.asChatCompletionResponse()._additionalProperties()["tps"] as JsonNumber).value as Float

        if (response == "") {
            //Empty content as Llama Stack is returning a tool call
            val toolCalls = result.asChatCompletionResponse().completionMessage().toolCalls()
            return if (toolCalls.isNotEmpty()) {
                functionDispatch(toolCalls, ctx)
            } else {
                "Empty tool calls and model response. File a bug"
            }
        }
        else {
            return response;
        }
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