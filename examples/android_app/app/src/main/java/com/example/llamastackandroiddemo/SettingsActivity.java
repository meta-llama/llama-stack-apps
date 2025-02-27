/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamastackandroiddemo;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import com.google.gson.Gson;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class SettingsActivity extends AppCompatActivity {

  private String mModelFilePath = "";
  private String mTokenizerFilePath = "";
  private TextView mBackendTextView;
  private TextView mModelTextView;
  private TextView mTokenizerTextView;
  private TextView mModelTypeTextView;
  private EditText mSystemPromptEditText;
  private Button mLoadModelButton;
  private double mSetTemperature;
  private String mSystemPrompt;
  private String mUserPrompt;
  private BackendType mBackendType;
  private ModelType mModelType;
  private EditText mRemoteURLEditText;
  private String mRemoteURL;
  public SettingsFields mSettingsFields;
  private String mRemoteModel;
  private TextView mRemoteModelTextView;

  private DemoSharedPreferences mDemoSharedPreferences;
  public static double TEMPERATURE_MIN_VALUE = 0.01;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_settings);
	  getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.status_bar));
	  getWindow().setNavigationBarColor(ContextCompat.getColor(this, R.color.nav_bar));
	  ViewCompat.setOnApplyWindowInsetsListener(
        requireViewById(R.id.main),
        (v, insets) -> {
          Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
          v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
          return insets;
        });
    mDemoSharedPreferences = new DemoSharedPreferences(getBaseContext());
    mSettingsFields = new SettingsFields();
    setupSettings();
  }

  private void setupSettings() {
    mBackendTextView = requireViewById(R.id.backendTextView);
    mModelTextView = requireViewById(R.id.modelTextView);
    mTokenizerTextView = requireViewById(R.id.tokenizerTextView);
    mModelTypeTextView = requireViewById(R.id.modelTypeTextView);
    ImageButton backendImageButton = requireViewById(R.id.backendImageButton);
    ImageButton modelImageButton = requireViewById(R.id.modelImageButton);
    ImageButton tokenizerImageButton = requireViewById(R.id.tokenizerImageButton);
    ImageButton modelTypeImageButton = requireViewById(R.id.modelTypeImageButton);
    mSystemPromptEditText = requireViewById(R.id.systemPromptText);
    mRemoteURLEditText = requireViewById(R.id.remoteURLEditText);

    loadSettings();

    // TODO: The two setOnClickListeners will be removed after file path issue is resolved
    backendImageButton.setOnClickListener(
        view -> {
          setupBackendSelectorDialog();
        });
    modelImageButton.setOnClickListener(
        view -> {
          setupModelSelectorDialog();
        });
    tokenizerImageButton.setOnClickListener(
        view -> {
          setupTokenizerSelectorDialog();
        });
    modelTypeImageButton.setOnClickListener(
        view -> {
          setupModelTypeSelectorDialog();
        });
    mModelFilePath = mSettingsFields.getModelFilePath();
    if (!mModelFilePath.isEmpty()) {
      mModelTextView.setText(getFilenameFromPath(mModelFilePath));
    }
    mTokenizerFilePath = mSettingsFields.getTokenizerFilePath();
    if (!mTokenizerFilePath.isEmpty()) {
      mTokenizerTextView.setText(getFilenameFromPath(mTokenizerFilePath));
    }
    mModelType = mSettingsFields.getModelType();
    AppLogging.getInstance().log("mModelType from settings " + mModelType);
    if (mModelType != null) {
      mModelTypeTextView.setText(mModelType.toString());
    }
    mBackendType = mSettingsFields.getBackendType();
    AppLogging.getInstance().log("mBackendType from settings " + mBackendType);
    if (mBackendType != null) {
      mBackendTextView.setText(mBackendType.toString());
      setBackendSettingMode();
    }

    setupParameterSettings();
    setupPromptSettings();
    setupClearChatHistoryButton();
    setupLoadModelButton();
    setupRemoteInferenceSettings();
  }

  private void setupLoadModelButton() {
    mLoadModelButton = requireViewById(R.id.loadModelButton);
    mLoadModelButton.setEnabled(true);
    mLoadModelButton.setOnClickListener(
        view -> {
          new AlertDialog.Builder(this)
              .setTitle("Load Model")
              .setMessage("Do you really want to load the new model?")
              .setIcon(android.R.drawable.ic_dialog_alert)
              .setPositiveButton(
                  android.R.string.yes,
                  new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                      mSettingsFields.saveLoadModelAction(true);
                      mLoadModelButton.setEnabled(false);
                      onBackPressed();
                    }
                  })
              .setNegativeButton(android.R.string.no, null)
              .show();
        });
  }

  private void setupClearChatHistoryButton() {
    Button clearChatButton = requireViewById(R.id.clearChatButton);
    clearChatButton.setOnClickListener(
        view -> {
          new AlertDialog.Builder(this)
              .setTitle("Delete Chat History")
              .setMessage("Do you really want to delete chat history?")
              .setIcon(android.R.drawable.ic_dialog_alert)
              .setPositiveButton(
                  android.R.string.yes,
                  new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                      mSettingsFields.saveIsClearChatHistory(true);
                    }
                  })
              .setNegativeButton(android.R.string.no, null)
              .show();
        });
  }

  private void setupParameterSettings() {
    setupTemperatureSettings();
  }

  private void setupTemperatureSettings() {
    mSetTemperature = mSettingsFields.getTemperature();
    EditText temperatureEditText = requireViewById(R.id.temperatureEditText);
    temperatureEditText.setText(String.valueOf(mSetTemperature));
    temperatureEditText.addTextChangedListener(
        new TextWatcher() {
          @Override
          public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

          @Override
          public void onTextChanged(CharSequence s, int start, int before, int count) {}

          @Override
          public void afterTextChanged(Editable s) {
            mSetTemperature = Double.parseDouble(s.toString());
            // This is needed because temperature is changed together with model loading
            // Once temperature is no longer in LlamaModule constructor, we can remove this
            mSettingsFields.saveLoadModelAction(true);
            saveSettings();
          }
        });
  }

  private void setupRemoteInferenceSettings() {
    mRemoteURL = mSettingsFields.getRemoteURL();
    AppLogging.getInstance().log("mRemoteURL from settings " + mRemoteURL);
    if (mRemoteURL != null) {
      mRemoteURLEditText.setText(mRemoteURL);
    }
    mRemoteURLEditText.addTextChangedListener(
            new TextWatcher() {
              @Override
              public void beforeTextChanged(CharSequence s, int start, int count, int after) {
              }

              @Override
              public void onTextChanged(CharSequence s, int start, int before, int count) {
              }

              @Override
              public void afterTextChanged(Editable s) {
                mRemoteURL = s.toString();
                AppLogging.getInstance().log("after text change remote url" + mRemoteURL);
                mSettingsFields.saveRemoteURL(mRemoteURL);
                saveSettings();
              }
            }
    );

    mRemoteModelTextView = requireViewById(R.id.remoteModelTextView);
    ImageButton mRemoteModelImageButton = requireViewById(R.id.remoteModelImageButton);

    mRemoteModel = mSettingsFields.getRemoteModel();
    AppLogging.getInstance().log("mRemoteModel from settings " + mRemoteModel);
    if (mRemoteModel != null) {
      mRemoteModelTextView.setText(mRemoteModel);
    }

    mRemoteModelImageButton.setOnClickListener(
            view -> {
              String[] models = ModelUtils.getSupportedRemoteModels().toArray(new String[0]);
              AlertDialog.Builder modelBuilder = new AlertDialog.Builder(this);
              modelBuilder.setTitle("Select remote model");
              modelBuilder.setSingleChoiceItems(
                      models,
                      -1,
                      (dialog, item) -> {
                        mRemoteModelTextView.setText(models[item]);
                        mRemoteModel = models[item];
                        dialog.dismiss();
                      });
              modelBuilder.create().show();
            });
  }

  private void setupPromptSettings() {
    setupSystemPromptSettings();
  }

  private void setupSystemPromptSettings() {
    mSystemPrompt = mSettingsFields.getSystemPrompt();
    mSystemPromptEditText.setText(mSystemPrompt);
    mSystemPromptEditText.addTextChangedListener(
        new TextWatcher() {
          @Override
          public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

          @Override
          public void onTextChanged(CharSequence s, int start, int before, int count) {}

          @Override
          public void afterTextChanged(Editable s) {
            mSystemPrompt = s.toString();
          }
        });

    ImageButton resetSystemPrompt = requireViewById(R.id.resetSystemPrompt);
    resetSystemPrompt.setOnClickListener(
        view -> {
          new AlertDialog.Builder(this)
              .setTitle("Reset System Prompt")
              .setMessage("Do you really want to reset system prompt?")
              .setIcon(android.R.drawable.ic_dialog_alert)
              .setPositiveButton(
                  android.R.string.yes,
                  new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                      // Clear the messageAdapter and sharedPreference
                      mSystemPromptEditText.setText(PromptFormat.DEFAULT_SYSTEM_PROMPT);
                    }
                  })
              .setNegativeButton(android.R.string.no, null)
              .show();
        });
  }

  private void setupBackendSelectorDialog() {
    // Convert enum to list
    List<String> backendTypesList = new ArrayList<>();
    for (BackendType backendType : BackendType.values()) {
      backendTypesList.add(backendType.toString());
    }
    // Alert dialog builder takes in arr of string instead of list
    String[] backendTypes = backendTypesList.toArray(new String[0]);
    AlertDialog.Builder backendTypeBuilder = new AlertDialog.Builder(this);
    backendTypeBuilder.setTitle("Select backend type");
    backendTypeBuilder.setSingleChoiceItems(
        backendTypes,
        -1,
        (dialog, item) -> {
          mBackendTextView.setText(backendTypes[item]);
          mBackendType = BackendType.valueOf(backendTypes[item]);
          setBackendSettingMode();
          dialog.dismiss();
        });

    backendTypeBuilder.create().show();
  }

  private void setupModelSelectorDialog() {
    String[] pteFiles = listLocalFile("/data/local/tmp/llama/", ".pte");
    AlertDialog.Builder modelPathBuilder = new AlertDialog.Builder(this);
    modelPathBuilder.setTitle("Select model path");

    modelPathBuilder.setSingleChoiceItems(
        pteFiles,
        -1,
        (dialog, item) -> {
          mModelFilePath = pteFiles[item];
          mModelTextView.setText(getFilenameFromPath(mModelFilePath));
          mLoadModelButton.setEnabled(true);
          dialog.dismiss();
        });

    modelPathBuilder.create().show();
  }

  private static String[] listLocalFile(String path, String suffix) {
    File directory = new File(path);
    if (directory.exists() && directory.isDirectory()) {
      File[] files = directory.listFiles((dir, name) -> name.toLowerCase().endsWith(suffix));
      String[] result = new String[files.length];
      for (int i = 0; i < files.length; i++) {
        if (files[i].isFile() && files[i].getName().endsWith(suffix)) {
          result[i] = files[i].getAbsolutePath();
        }
      }
      return result;
    }
    return new String[] {};
  }

  private void setupModelTypeSelectorDialog() {
    // Convert enum to list
    List<String> modelTypesList = new ArrayList<>();
    for (ModelType modelType : ModelType.values()) {
      modelTypesList.add(modelType.toString());
    }
    // Alert dialog builder takes in arr of string instead of list
    String[] modelTypes = modelTypesList.toArray(new String[0]);
    AlertDialog.Builder modelTypeBuilder = new AlertDialog.Builder(this);
    modelTypeBuilder.setTitle("Select model type");
    modelTypeBuilder.setSingleChoiceItems(
        modelTypes,
        -1,
        (dialog, item) -> {
          mModelTypeTextView.setText(modelTypes[item]);
          mModelType = ModelType.valueOf(modelTypes[item]);
          dialog.dismiss();
        });

    modelTypeBuilder.create().show();
  }

  private void setupTokenizerSelectorDialog() {
    String[] binFiles = listLocalFile("/data/local/tmp/llama/", ".bin");
    String[] modelFiles = listLocalFile("/data/local/tmp/llama/", ".model");
    String[] tokenizerFiles = new String[binFiles.length + modelFiles.length];
    System.arraycopy(binFiles, 0, tokenizerFiles, 0, binFiles.length);
    System.arraycopy(modelFiles, 0, tokenizerFiles, binFiles.length, modelFiles.length);
    AlertDialog.Builder tokenizerPathBuilder = new AlertDialog.Builder(this);
    tokenizerPathBuilder.setTitle("Select tokenizer path");
    tokenizerPathBuilder.setSingleChoiceItems(
        tokenizerFiles,
        -1,
        (dialog, item) -> {
          mTokenizerFilePath = tokenizerFiles[item];
          mTokenizerTextView.setText(getFilenameFromPath(mTokenizerFilePath));
          mLoadModelButton.setEnabled(true);
          dialog.dismiss();
        });

    tokenizerPathBuilder.create().show();
  }

  private String getFilenameFromPath(String uriFilePath) {
    String[] segments = uriFilePath.split("/");
    if (segments.length > 0) {
      return segments[segments.length - 1]; // get last element (aka filename)
    }
    return "";
  }

  private void setBackendSettingMode() {
    if (mBackendType.equals(BackendType.XNNPACK)) {
      setXNNPACKSettingMode();
    }
  }

  private void setXNNPACKSettingMode() {
    requireViewById(R.id.modelLayout).setVisibility(View.VISIBLE);
    requireViewById(R.id.tokenizerLayout).setVisibility(View.VISIBLE);
    requireViewById(R.id.parametersView).setVisibility(View.VISIBLE);
    requireViewById(R.id.temperatureLayout).setVisibility(View.VISIBLE);
  }

  private void setMediaTekSettingMode() {
    requireViewById(R.id.modelLayout).setVisibility(View.GONE);
    requireViewById(R.id.tokenizerLayout).setVisibility(View.GONE);
    requireViewById(R.id.parametersView).setVisibility(View.GONE);
    requireViewById(R.id.temperatureLayout).setVisibility(View.GONE);
    mModelFilePath = "/in/mtk/llama/runner";
    mTokenizerFilePath = "/in/mtk/llama/runner";
  }

  private void loadSettings() {
    Gson gson = new Gson();
    String settingsFieldsJSON = mDemoSharedPreferences.getSettings();
    if (!settingsFieldsJSON.isEmpty()) {
      AppLogging.getInstance().log("mSettingsFields " + settingsFieldsJSON);
      mSettingsFields = gson.fromJson(settingsFieldsJSON, SettingsFields.class);
    }
  }

  private void saveSettings() {
    AppLogging.getInstance().log("saving settings " +mRemoteURL + mModelFilePath);
    mSettingsFields.saveModelPath(mModelFilePath);
    mSettingsFields.saveTokenizerPath(mTokenizerFilePath);
    mSettingsFields.saveParameters(mSetTemperature);
    mSettingsFields.savePrompts(mSystemPrompt);
    mSettingsFields.saveModelType(mModelType);
    mSettingsFields.saveBackendType(mBackendType);
    mSettingsFields.saveRemoteURL(mRemoteURL);
    mSettingsFields.saveRemoteModel(mRemoteModel);
    mDemoSharedPreferences.addSettings(mSettingsFields);
  }

  @Override
  public void onBackPressed() {
    super.onBackPressed();
    saveSettings();
  }
}
