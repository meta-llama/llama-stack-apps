package com.example.llamastackandroiddemo

class AvailableFunctions private constructor() {
    private val dictionary: MutableMap<String, String> = HashMap()

    companion object {
        @Volatile private var instance: AvailableFunctions? = null
        fun getInstance(): AvailableFunctions {
            return instance ?: synchronized(this) {
                instance ?: AvailableFunctions().also { instance = it }
            }
        }
    }
    init {
        // Initialize the dictionary here
        val functionName = "createCalendarEvent"
        val functionDefinition = """
                    Use this function only if user has intention to schedule a calendar event as a note or a reminder
                    {
                        "name": "createCalendarEvent",
                        "description": "Add a event to the system default Calendar app in Android that include title, description, start and end time. Note that the provided function is in Kotlin syntax.",
                        "parameters": {
                            "type": "dict",
                            "required": [
                                "title",
                                "description",
                                "startDate",
                                "endDate",
                                "startTime",
                                "endTime"
                            ],
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "The title for the new Calendar event",
                                    "default": "none"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "The description for the new Calendar event",
                                    "default": "none"
                                },
                                "startDate": {
                                    "type": "string",
                                    "description": "The start Date of the Calendar event in format of yyyy-MM-dd",
                                    "default": "none"
                                },
                                "endDate": {
                                    "type": "string",
                                    "description": "The end Date of the Calendar event in format of yyyy-MM-dd",
                                    "default": "none"
                                },
                                "startTime": {
                                    "type": "string",
                                    "description": "The start time of the Calendar event in format of HH:mm",
                                    "default": "none"
                                },
                                "endTime": {
                                    "type": "string",
                                    "description": "The end time of the Calendar event in format of HH:mm",
                                    "default": "none"
                                }
                            }
                        }
                    }
                """

        put(functionName, functionDefinition)
    }


    fun put(key: String, value: String) {
        dictionary[key] = value
    }

    fun get(key: String): String? {
        return dictionary[key]
    }

    fun remove(key: String): String? {
        return dictionary.remove(key)
    }

    fun containsKey(key: String): Boolean {
        return dictionary.containsKey(key)
    }

    fun keys(): List<String> {
        return dictionary.keys.toList()
    }

    fun values(): List<String> {
        return dictionary.values.toList()
    }
}