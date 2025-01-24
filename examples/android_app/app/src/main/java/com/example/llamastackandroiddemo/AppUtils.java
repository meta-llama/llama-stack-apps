package com.example.llamastackandroiddemo;

public class AppUtils {
	// Generation Mode
	public static final String REMOTE = "Remote";
	public static final String LOCAL = "Local";
	public static final int CONVERSATION_HISTORY_MESSAGE_LOOKBACK = 3;

	public static String getDefaultGenerationMode() {
		return LOCAL;
	}
}
