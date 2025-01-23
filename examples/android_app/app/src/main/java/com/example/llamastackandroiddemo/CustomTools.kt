package com.example.llamastackandroiddemo

import com.llama.llamastack.models.ToolDef

class CustomTools {
    companion object {
        fun getCreateCalendarEventTool(): ToolDef {
            val createCalendarEventTool =
                ToolDef.builder()
                    .description("Add a event to the system default Calendar app in Android that include title, description, start and end time. Note that the provided function is in Kotlin syntax.")
                    .name("createCalendarEvent")
                    .parameters(
                        listOf(
                            ToolDef.Parameter.builder()
                                .description("The title for the new Calendar event")
                                .name("title")
                                .parameterType("string")
                                .required(true)
                                .default(null)
                                .build(),
                            ToolDef.Parameter.builder()
                                .description("The description for the new Calendar event")
                                .name("description")
                                .parameterType("string")
                                .required(true)
                                .default(null)
                                .build(),
                            ToolDef.Parameter.builder()
                                .description("The start Date of the Calendar event in format of yyyy-MM-dd")
                                .name("startDate")
                                .parameterType("string")
                                .required(true)
                                .default(null)
                                .build(),
                            ToolDef.Parameter.builder()
                                .description("The end Date of the Calendar event in format of yyyy-MM-dd")
                                .name("endDate")
                                .parameterType("string")
                                .required(true)
                                .default(null)
                                .build(),
                            ToolDef.Parameter.builder()
                                .description("The start time of the Calendar event in format of HH:mm")
                                .name("startTime")
                                .parameterType("string")
                                .required(true)
                                .default(null)
                                .build(),
                            ToolDef.Parameter.builder()
                                .description("The end time of the Calendar event in format of HH:mm")
                                .name("endTime")
                                .parameterType("string")
                                .required(true)
                                .default(null)
                                .build()
                        )
                    )
                    .build()

            return createCalendarEventTool
        }
    }
}