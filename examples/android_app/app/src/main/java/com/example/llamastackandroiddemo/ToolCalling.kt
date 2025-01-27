package com.example.llamastackandroiddemo

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.ContentResolver
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.provider.CalendarContract
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.llama.llamastack.models.CompletionMessage
import com.llama.llamastack.models.ContentDelta
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.TimeZone

//Tool calling helper functions

fun functionDispatchNotWorking(toolCalls:List<ContentDelta.ToolCallDelta>, ctx: Context): String {
    return "0"
//    var response = ""
//
//
//    for (toolCall in toolCalls) {
//        val toolName = toolCall.toolName().toString()
//        val properties = toolCall.arguments()._additionalProperties()
//        response += when (toolName) {
//            "createCalendarEvent" -> createCalendarEvent(
//                properties["title"].toString(),
//                properties["description"].toString(),
//                properties["startDate"].toString(),
//                properties["endDate"].toString(),
//                properties["startTime"].toString(),
//                properties["endTime"].toString(),
//                ctx
//            )
//            else -> "Function in registry but execution is not implemented. Add your function in the AvailableFunctions.kt"
//        } + "\n"
//    }
//    return if(response.isEmpty()){
//        "Function is not registered and cannot be recognized. Please Add your function in the AvailableFunctions.kt and provide implementation"
//    } else{
//        // remove hanging "\n"
//        response.removeSuffix("\n")
//    }
}

fun functionDispatch(toolCalls:List<CompletionMessage.ToolCall>, ctx: Context): String {
    var response = ""


    for (toolCall in toolCalls) {
        val toolName = toolCall.toolName().toString()
        val properties = toolCall.arguments()._additionalProperties()
            response += when (toolName) {
                "createCalendarEvent" -> createCalendarEvent(
                    properties["title"].toString(),
                    properties["description"].toString(),
                    properties["startDate"].toString(),
                    properties["endDate"].toString(),
                    properties["startTime"].toString(),
                    properties["endTime"].toString(),
                    ctx
                )
                else -> "Function in registry but execution is not implemented. Add your function in the AvailableFunctions.kt"
            } + "\n"
    }
    return if(response.isEmpty()){
        "Function is not registered and cannot be recognized. Please Add your function in the AvailableFunctions.kt and provide implementation"
    } else{
        // remove hanging "\n"
        response.removeSuffix("\n")
    }
}

private fun createCalendarEvent(title:String, description: String, startDate:String, endDate:String, startTime:String, endTime:String, ctx: Context): String {
    val calendarPermissionRequestCode = 101

    if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.WRITE_CALENDAR) == PackageManager.PERMISSION_GRANTED) {
        //Convert Strings to Int for the time
        val (startYear, startMonth, startDay) = startDate.split("-").map { it.toInt() }
        val (startHour, startMinute) = startTime.split(":").map { it.toInt() }
        val (endYear, endMonth, endDay) = endDate.split("-").map { it.toInt() }
        val (endHour, endMinute) = endTime.split(":").map { it.toInt() }

        val calendarStartTime = Calendar.getInstance().apply {
            set(startYear, startMonth - 1, startDay, startHour, startMinute) // Year, Month - 1 (i.e. January is 0), Day, Hour, Minute
        }.timeInMillis

        val calendarEndTime = Calendar.getInstance().apply {
            set(endYear, endMonth - 1, endDay, endHour, endMinute)
        }.timeInMillis
        return addEventToCalendar(title, description, calendarStartTime, calendarEndTime, ctx)
    } else {
        // Permission is not granted. Request it.
        ActivityCompat.requestPermissions(ctx as Activity, arrayOf(Manifest.permission.WRITE_CALENDAR), calendarPermissionRequestCode)
        return "Calender App permission granted. Please try again."
    }
}

@SuppressLint("SimpleDateFormat")
private fun addEventToCalendar(title: String, description: String, startTime: Long, endTime: Long, ctx: Context) : String {
    val values = ContentValues().apply {
        put(CalendarContract.Events.CALENDAR_ID, 1) // Default calendar ID
        put(CalendarContract.Events.TITLE, title)
        put(CalendarContract.Events.DESCRIPTION, description)
        put(CalendarContract.Events.DTSTART, startTime)
        put(CalendarContract.Events.DTEND, endTime)
        put(CalendarContract.Events.EVENT_TIMEZONE, TimeZone.getDefault().id)
    }
    val resolver: ContentResolver = ctx.contentResolver
    val uri = resolver.insert(CalendarContract.Events.CONTENT_URI, values)
    if (uri != null) {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm")
        return "Calendar event - $title scheduled successfully starting at ${dateFormat.format(
            Date(startTime)
        )}, ends at ${dateFormat.format(
            Date(endTime)
        )}"
    } else {
        return "Calendar event failed to schedule"
    }
}