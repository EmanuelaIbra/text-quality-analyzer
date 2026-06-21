const API_URL = " ";

function onOpen() {
  DocumentApp.getUi()
    .createMenu("AI Writing Assistant")
    .addItem("Open Assistant", "showSidebar")
    .addToUi();
}

function showSidebar() {
  const html = HtmlService
    .createHtmlOutputFromFile("Sidebar")
    .setTitle("AI Writing Assistant");

  DocumentApp.getUi().showSidebar(html);
}

function getSelectedText() {
  const selection = DocumentApp.getActiveDocument().getSelection();

  if (!selection) {
    return {
      success: false,
      message: "Please select text first."
    };
  }

  const rangeElements = selection.getRangeElements();
  let selectedText = "";

  rangeElements.forEach(function(rangeElement) {
    const element = rangeElement.getElement();

    if (element.editAsText) {
      const textElement = element.asText();

      if (rangeElement.isPartial()) {
        const start = rangeElement.getStartOffset();
        const end = rangeElement.getEndOffsetInclusive();
        selectedText += textElement.getText().substring(start, end + 1);
      } else {
        selectedText += textElement.getText();
      }

      selectedText += "\n";
    }
  });

  return {
    success: true,
    text: selectedText.trim()
  };
}

function analyzeAndRewriteSelectedText(mode) {
  const selected = getSelectedText();

  if (!selected.success) {
    return selected;
  }

  const response = UrlFetchApp.fetch(API_URL, {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify({
      text: selected.text,
      mode: mode || "concise"
    }),
    muteHttpExceptions: true
  });

  const status = response.getResponseCode();
  const body = response.getContentText();

  if (status !== 200) {
    return {
      success: false,
      message: body
    };
  }

  const result = JSON.parse(body);

  PropertiesService.getDocumentProperties().setProperty(
    "latestOptimizedText",
    result.final
  );

  return {
    success: true,
    original: result.original,
    grammar_corrected: result.grammar_corrected,
    rewritten: result.rewritten,
    final: result.final,
    metrics: result.metrics,
    repetition_analysis: result.repetition_analysis,
    redundancy_report: result.redundancy_report
  };
}

function acceptRewrite() {
  const optimizedText =
    PropertiesService.getDocumentProperties().getProperty(
      "latestOptimizedText"
    );

  if (!optimizedText) {
    return {
      success: false,
      message: "No optimized text available."
    };
  }

  const selection = DocumentApp.getActiveDocument().getSelection();

  if (!selection) {
    return {
      success: false,
      message: "Please select the original text again."
    };
  }

  const rangeElements = selection.getRangeElements();
  const firstRangeElement = rangeElements[0];
  const firstElement = firstRangeElement.getElement().asText();

  if (firstRangeElement.isPartial()) {
    const start = firstRangeElement.getStartOffset();
    const end = firstRangeElement.getEndOffsetInclusive();

    firstElement.deleteText(start, end);
    firstElement.insertText(start, optimizedText);
  } else {
    firstElement.setText(optimizedText);
  }

  return {
    success: true,
    message: "Text replaced successfully."
  };
}