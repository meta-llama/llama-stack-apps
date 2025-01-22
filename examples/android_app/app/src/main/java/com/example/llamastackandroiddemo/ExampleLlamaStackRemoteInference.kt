package com.example.llamastackandroiddemo

import android.content.Context
import android.util.Log
import com.llama.llamastack.client.LlamaStackClientClient
import com.llama.llamastack.client.okhttp.LlamaStackClientOkHttpClient
import com.llama.llamastack.models.AgentConfig
import com.llama.llamastack.models.AgentCreateParams
import com.llama.llamastack.models.AgentSessionCreateParams
import com.llama.llamastack.models.AgentTurnCreateParams
import com.llama.llamastack.models.InferenceChatCompletionParams
import com.llama.llamastack.models.InterleavedContent
import com.llama.llamastack.models.SamplingParams
import com.llama.llamastack.models.SystemMessage
import com.llama.llamastack.models.ToolResponseMessage
import com.llama.llamastack.models.UserMessage
import com.llama.llamastack.services.blocking.agents.TurnService
import kotlinx.datetime.Clock
import java.time.Instant
import java.time.ZoneId
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.CompletableFuture

interface InferenceStreamingCallback {
    fun onStreamReceived(message: String)
    fun onStatStreamReceived(tps: Float)
}

class ExampleLlamaStackRemoteInference(remoteURL: String) {

    public var client: LlamaStackClientClient? = null

    private var agentConfig: AgentConfig? = null

    init {
        try {
            client = LlamaStackClientOkHttpClient
                .builder()
                .baseUrl(remoteURL)
                .headers(mapOf("x-llamastack-client-version" to listOf("0.1.0")))
                .build()
        } catch (e: Exception) {
            client = null
            AppLogging.getInstance().log(e.message)
        }
    }

    fun inferenceStartWithoutAgent(modelName: String, temperature: Double, prompt: ArrayList<Message>, systemPrompt:String, ctx: Context): String {
        val future = CompletableFuture<String>()
        val thread = Thread {
            try {
                val response = inferenceCallWithoutAgent(modelName, temperature, prompt, systemPrompt, ctx, true);
                future.complete(response)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        thread.start();
        return future.get();
    }

    fun inferenceStartWithAgent(agentId: String, sessionId: String, turnService: TurnService, prompt: ArrayList<Message>, ctx: Context): String {
        val future = CompletableFuture<String>()
        val thread = Thread {
            try {
                val response = remoteAgentInference(agentId, sessionId, turnService, prompt, ctx)
                future.complete(response)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        thread.start();
        return future.get();
    }



    //Example running simple inference + tool calls without using agent's workflow
    private fun inferenceCallWithoutAgent(modelName: String, temperature: Double, conversationHistory: ArrayList<Message>, systemPrompt: String, ctx: Context, streaming: Boolean): String {
        if (client == null) {
            AppLogging.getInstance().log("client is null for remote inference");
            return "[ERROR] client is null for remote inference"
        }

        //Get the current time in ISO format and pass it to the model in system prompt as a reference. This is useful for any scheduling and vague timing reference from user prompt.
        val zdt = ZonedDateTime.ofInstant(Instant.parse(Clock.System.now().toString()), ZoneId.systemDefault())
        val formattedZdt = zdt.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"))
        val availableFunctions = AvailableFunctions.getInstance()
        val functionDefinitions = availableFunctions.values()
        var instruction = systemPrompt
        //If no System prompt configured by the user, use default tool call system prompt
        if (instruction == "") {
            instruction = """
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

        var response = ""
        try {
            if (streaming) {
                val result = client!!.inference().chatCompletionStreaming(
                    InferenceChatCompletionParams.builder()
                        .modelId(modelName)
                        .samplingParams(
                            SamplingParams.builder()
                                .strategy(
                                    SamplingParams.Strategy.ofGreedySamplingStrategy(
                                        SamplingParams.Strategy.GreedySamplingStrategy.builder()
                                            .type(
                                                SamplingParams.Strategy.GreedySamplingStrategy.Type
                                                    .GREEDY
                                        ).build()
                                    )
                                ).build()
                        )
                        .messages(
                            constructLSMessagesFromConversationHistoryAndSystemPrompt(conversationHistory, instruction)
                        )
                        .build()
                )
                val callback = ctx as InferenceStreamingCallback
                result.use {
                    result.asSequence().forEach {
                        val delta = it.asChatCompletionResponseStreamChunk().event().delta()
                        if (delta.isToolCallDelta()) {
                            val toolCall = delta.toolCallDelta()?.content()?.toolCall()
                            if (toolCall != null) {
                                callback.onStreamReceived("\n" + functionDispatch(listOf(toolCall), ctx))
                            } else {
                                callback.onStreamReceived("\n" + "Empty tool call. File a bug")
                            }

                        }
                        if (it.asChatCompletionResponseStreamChunk().event().stopReason().toString() != "end_of_turn") {
                            callback.onStreamReceived(it.asChatCompletionResponseStreamChunk().event().delta().textDelta()?.text().toString())
                        }
                    }
                }
            } else {
                val result = client!!.inference().chatCompletion(
                    InferenceChatCompletionParams.builder()
                        .modelId(modelName)
                        .samplingParams(
                            SamplingParams.builder()
                                .strategy(
                                    SamplingParams.Strategy.ofGreedySamplingStrategy(
                                        SamplingParams.Strategy.GreedySamplingStrategy.builder()
                                            .type(
                                                SamplingParams.Strategy.GreedySamplingStrategy.Type
                                                    .GREEDY
                                            ).build()
                                    )
                                ).build()
                        )
                        .messages(
                            constructLSMessagesFromConversationHistoryAndSystemPrompt(conversationHistory, instruction)
                        )
                        .build()
                )
                response = result.asChatCompletionResponse().completionMessage().content().string().toString();
                if (response == "") {
                    //Empty content as Llama Stack is returning a tool call in non-streaming mode
                    val toolCalls = result.asChatCompletionResponse().completionMessage().toolCalls()
                    return if (toolCalls.isNotEmpty()) {
                        functionDispatch(toolCalls, ctx)
                    } else {
                        "Empty tool calls and model response. File a bug"
                    }
                }
            }
        } catch (e : Exception) {
            AppLogging.getInstance().log("Exception on remote inference " + e.message);
            return "Exception on remote inference " + e.message
        }
        return response
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
                    .content(InterleavedContent.ofString(systemPrompt))
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
                        .content(InterleavedContent.ofString(chat.text))
                        .role(UserMessage.Role.USER)
                        .build()
                )
            } else {
                // Assistant message (aka previous prompt response)
                inferenceMessage = InferenceChatCompletionParams.Message.ofCompletionMessage(
                    InferenceChatCompletionParams.Message.CompletionMessage.builder()
                        .content(InterleavedContent.ofString(chat.text))
                        .stopReason(InferenceChatCompletionParams.Message.CompletionMessage.StopReason.END_OF_MESSAGE)
                        .toolCalls(emptyList())
                        .role(InferenceChatCompletionParams.Message.CompletionMessage.Role.ASSISTANT)
                        .build()
                )
            }
            messageList.add(inferenceMessage)
        }
        AppLogging.getInstance().log("conversation history length "  + messageList.size)
        return messageList
    }


    fun createRemoteAgent(modelName: String, temperature: Double, systemPrompt: String, ctx: Context): Triple<String, String, TurnService> {
        val agentConfig = createRemoteAgentConfig(modelName, temperature, systemPrompt)
        val agentService = client!!.agents()
        val agentCreateResponse = agentService.create(
            AgentCreateParams.builder()
                .agentConfig(agentConfig)
//                .xLlamaStackClientVersion("X-LlamaStack-Client-Version")
//                .xLlamaStackProviderData("X-LlamaStack-ProviderData")
                .build(),
        )

        val agentId = agentCreateResponse.agentId()
        val sessionService = agentService.session()
        val agentSessionCreateResponse = sessionService.create(
            AgentSessionCreateParams.builder()
                .agentId(agentId)
                .sessionName("test-session")
//                .xLlamaStackClientVersion("X-LlamaStack-Client-Version")
//                .xLlamaStackProviderData("X-LlamaStack-ProviderData")
                .build()
        )

        val sessionId = agentSessionCreateResponse.sessionId()
        val turnService = agentService.turn()

        Log.d("Chester", "Agent created. Id = $agentId with Session = $sessionId")
        return Triple(agentId, sessionId, turnService)
    }

    //Example of running inference with customize tool calls using agent workflow.
    //Note Agent inference only support streaming at the moment.
    private fun remoteAgentInference(agentId: String, sessionId: String, turnService: TurnService, conversationHistory: ArrayList<Message>, ctx: Context): String {
        val agentTurnCreateResponseStream =
            turnService.createStreaming(
                AgentTurnCreateParams.builder()
                    .agentId(agentId)
                    .messages(
                        constructMessagesForAgent(conversationHistory)
                    )
                    .sessionId(sessionId)
//                    .documents(
//                        listOf(
//                            AgentTurnCreateParams.Document.builder()
//                                .content(AgentTurnCreateParams.Document.Content.ofString("string"))
//                                .mimeType("mime_type")
//                                .build()
//                        )
//                    )
//                    .toolgroups(listOf(AgentTurnCreateParams.Toolgroup.ofString("string")))
//                    .xLlamaStackClientVersion("X-LlamaStack-Client-Version")
//                    .xLlamaStackProviderData("X-LlamaStack-Provider-Data")
                    .build()
            )
        val callback = ctx as InferenceStreamingCallback
        agentTurnCreateResponseStream.use {
            agentTurnCreateResponseStream.asSequence().forEach {
                val agentResponsePayload = it.agentTurnResponseStreamChunk()?.event()?.payload()
                if (agentResponsePayload != null) {
                    when {
                        agentResponsePayload.isAgentTurnResponseTurnStartPayload() -> {
                            // Handle Turn Start Payload
                        }

                        agentResponsePayload.isAgentTurnResponseStepStartPayload() -> {
                            // Handle Step Start Payload

                            //Need-To-Fix: AgentTurnResponseStepStartPayload type mismatch bug.
                            val result = agentResponsePayload.agentTurnResponseStepStartPayload()?._additionalProperties()?.get("delta")?.asObject()?.get("text")
                            //We shouldn't need to do this if the payload is been correctly setup on the server side as StepProgressPayload
                            val toolcallJson = agentResponsePayload.agentTurnResponseStepStartPayload()?._additionalProperties()?.get("delta")?.asObject()?.get("content")
                            if (toolcallJson != null) {
                                val call_id = toolcallJson.asObject()?.get("call_id")
                                val tool_name = toolcallJson.asObject()?.get("tool_name")
                                val arguments = toolcallJson.asObject()?.get("arguments")
                                Log.d("Chester", "call_id: $call_id, tool_name: $tool_name, arguments: $arguments")

                            }
                            if (result != null) {
                                callback.onStreamReceived(result.toString())
                            }
                        }

                        agentResponsePayload.isAgentTurnResponseStepProgressPayload() -> {
                            // Handle Step Progress Payload
            //                        val result = agentResponsePayload.agentTurnResponseStepProgressPayload()?.delta().toolCallDelta()?.content().toolCall()
                        }

                        agentResponsePayload.isAgentTurnResponseStepCompletePayload() -> {
                            // Handle Step Complete Payload
                        }

                        agentResponsePayload.isAgentTurnResponseTurnCompletePayload() -> {
                            // Handle Turn Complete Payload
                        }
                    }
                }
                Log.d("Chester", "Streaming Agent responses: ${it.agentTurnResponseStreamChunk()?.event()}")
            }
        }
        return ""
    }

    private fun createRemoteAgentConfig(modelName: String, temperature: Double, systemPrompt: String): AgentConfig {
        //Get the current time in ISO format and pass it to the model in system prompt as a reference. This is useful for any scheduling and vague timing reference from user prompt.
        val zdt = ZonedDateTime.ofInstant(Instant.parse(Clock.System.now().toString()), ZoneId.systemDefault())
        //This should be replaced with Agent getting date and time with search tool
        val formattedZdt = zdt.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"))
        var instruction = systemPrompt
        //If no System prompt configured by the user, use default tool call system prompt
        if (instruction == "") {
            instruction = """
                            You are a helpful assistant that can also compose functions.
                            When user is asking a question that requires your reasoning or general chat, you should NOT generate functions.
                            
                            Today Date: $formattedZdt
                            
                            - Only function call if user's intention matches the function that you have access to.
                            - When looking for real time information use relevant functions if available.
                            - Ignore previous conversation history if you are making a tool call.
                                                                              
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

        val agentConfig =
            AgentConfig.builder()
                .enableSessionPersistence(false)
                .instructions(instruction)
                .maxInferIters(100)
                .model(modelName)
                .samplingParams(
                    SamplingParams.builder()
                        .strategy(
                            SamplingParams.Strategy.ofGreedySamplingStrategy(
                                SamplingParams.Strategy.GreedySamplingStrategy.builder()
                                    .type(SamplingParams.Strategy.GreedySamplingStrategy.Type.GREEDY)
                                    .build()
                            )
                        )
                        .build()
                )
                .toolChoice(AgentConfig.ToolChoice.AUTO)
                .toolPromptFormat(AgentConfig.ToolPromptFormat.PYTHON_LIST)
                .clientTools(
                    listOf(
                        CustomTools.getCreateCalendarEventTool()
                    )
                )
                .build()

        return agentConfig
    }

    private fun constructMessagesForAgent(
        conversationHistory: ArrayList<Message>,
    ):List<AgentTurnCreateParams.Message> {
        val messageList = ArrayList<AgentTurnCreateParams.Message>();

        // User and assistant messages
        for (chat in conversationHistory) {
            var inferenceMessage: AgentTurnCreateParams.Message
            if (chat.isSent) {
                // User message
                inferenceMessage = AgentTurnCreateParams.Message.ofUserMessage(
                    UserMessage.builder()
                        .content(InterleavedContent.ofString(chat.text))
                        .role(UserMessage.Role.USER)
                        .build()
                )
            } else {
                // Assistant message (aka previous prompt response)
                inferenceMessage = AgentTurnCreateParams.Message.ofToolResponseMessage(
                    ToolResponseMessage.builder()
                        .callId("")
                        .content(InterleavedContent.ofString(chat.text))
                        .role(ToolResponseMessage.Role.TOOL)
                        .toolName("")
                        .build()
                )
            }
            messageList.add(inferenceMessage)
        }
        AppLogging.getInstance().log("conversation history length "  + messageList.size)
        return messageList
    }
}