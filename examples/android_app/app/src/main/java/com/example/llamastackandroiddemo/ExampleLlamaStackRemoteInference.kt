package com.example.llamastackandroiddemo

import android.content.Context
import com.llama.llamastack.client.LlamaStackClientClient
import com.llama.llamastack.client.okhttp.LlamaStackClientOkHttpClient
import com.llama.llamastack.models.CompletionMessage
import com.llama.llamastack.models.InferenceChatCompletionParams
import com.llama.llamastack.models.SamplingParams
import com.llama.llamastack.models.SystemMessage
import com.llama.llamastack.models.UserMessage
import kotlinx.datetime.Clock
import java.time.Instant
import java.time.ZoneId
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.CompletableFuture

interface InferenceStreamingCallback {
    fun onStreamReceived(message: String)
}

class ExampleLlamaStackRemoteInference(remoteURL: String) {

    private var client: LlamaStackClientClient? = null

    init {
        try {
            client = LlamaStackClientOkHttpClient
                .builder()
                .baseUrl(remoteURL)
                .build()
        } catch (e: Exception) {
            client = null
            AppLogging.getInstance().log(e.message)
        }
    }

    fun inferenceStart(modelName: String, temperature: Double, prompt: ArrayList<Message>, systemPrompt:String, ctx: Context): String {
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
                val response = inferenceCall(modelName, temperature, prompt, sysPrompt, ctx, true);
                future.complete(response)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        thread.start();
        return future.get();

    }

    private fun inferenceCall(modelName: String, temperature: Double, conversationHistory: ArrayList<Message>, systemPrompt: String, ctx: Context, streaming: Boolean): String {
        if (client == null) {
            AppLogging.getInstance().log("client is null for remote inference");
            return "[ERROR] client is null for remote inference"
        }

        var response = ""
        try {
            if (streaming) {
                val result = client!!.inference().chatCompletionStreaming(
                    InferenceChatCompletionParams.builder()
                        .modelId(modelName)
                        .samplingParams(
                            SamplingParams.builder()
                                .temperature(temperature)
                                .build()
                        )
                        .messages(
                            constructLSMessagesFromConversationHistoryAndSystemPrompt(conversationHistory, systemPrompt)
                        )
                        .build()
                )
                val callback = ctx as InferenceStreamingCallback
                result.use {
                    result.asSequence().forEach {
                        println(it)
                        callback.onStreamReceived(it.asChatCompletionResponseStreamChunk().event().delta().string().toString())
                    }
                }
            } else {
                val result = client!!.inference().chatCompletion(
                    InferenceChatCompletionParams.builder()
                        .modelId(modelName)
                        .samplingParams(
                            SamplingParams.builder()
                                .temperature(temperature)
                                .build()
                        )
                        .messages(
                            constructLSMessagesFromConversationHistoryAndSystemPrompt(conversationHistory, systemPrompt)
                        )
                        .build()
                )
                response = result.asChatCompletionResponse().completionMessage().content().string().toString();
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
        } catch (e : Exception) {
            AppLogging.getInstance().log("Exception on remote inference " + e.message);
            response = e.message.toString() + ". Check if your remote URL is accessible."
        }
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