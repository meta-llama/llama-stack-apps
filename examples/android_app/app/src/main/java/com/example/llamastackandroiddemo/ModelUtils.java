/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamastackandroiddemo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ModelUtils {
  // XNNPACK or QNN
  static final int TEXT_MODEL = 1;

  // XNNPACK
  static final int VISION_MODEL = 2;
  static final int VISION_MODEL_IMAGE_CHANNELS = 3;
  static final int VISION_MODEL_SEQ_LEN = 768;
  static final int TEXT_MODEL_SEQ_LEN = 256;

  // MediaTek
  static final int MEDIATEK_TEXT_MODEL = 3;

  public static int getModelCategory(ModelType modelType, BackendType backendType) {
    if (backendType.equals(BackendType.XNNPACK)) {
      switch (modelType) {
        case LLAMA_3:
        case LLAMA_3_1:
        case LLAMA_3_2:
        default:
          return TEXT_MODEL;
      }
    }

    return TEXT_MODEL; // default
  }

  public static List<String> getSupportedRemoteModels() {
    return Arrays.asList(
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct"
            );
  }

  public static int getSequenceLengthForPrompt(String userPrompt, String systemPrompt) {
    // Using the logic of 1 char = 0.75 token, + 64 buffer
    return (int)((userPrompt.length() + systemPrompt.length()) * 0.75) + 64;
  }

  public static int getSequenceLengthForConversationHistory(ArrayList<Message> messages, String systemPrompt) {
    int seq_len = 0;
    for (Message message: messages) {
      seq_len += message.getText().length();
    }
    seq_len += systemPrompt.length();

    return seq_len + 128; // 64 is the buffer
  }
}
