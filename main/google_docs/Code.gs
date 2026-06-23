const BASE_API_URL = "https://negotiate-resident-swept.ngrok-free.dev";

function onOpen() {
  DocumentApp.getUi()
    .createMenu("Writing Assistant")
    .addItem("Open Assistant", "showSidebar")
    .addToUi();
}

function showSidebar() {
  const html = HtmlService
    .createHtmlOutputFromFile("Sidebar")
    .setTitle("Writing Assistant");

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

        selectedText += textElement
          .getText()
          .substring(start, end + 1);
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

function analyzeSelectedText() {
  const selected = getSelectedText();

  if (!selected.success) {
    return selected;
  }

  const response = UrlFetchApp.fetch(BASE_API_URL + "/analyze", {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify({
      text: selected.text
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
    "latestOriginalText",
    selected.text
  );

  PropertiesService.getDocumentProperties().setProperty(
    "latestAnalysisResult",
    JSON.stringify(result)
  );

  return result;
}

function rewriteAfterReview(mode, decisions) {
  const originalText =
    PropertiesService.getDocumentProperties().getProperty(
      "latestOriginalText"
    );

  if (!originalText) {
    return {
      success: false,
      message: "No analyzed text found. Please run analysis first."
    };
  }

  const response = UrlFetchApp.fetch(BASE_API_URL + "/rewrite", {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify({
      text: originalText,
      mode: mode || "concise",
      decisions: decisions || {}
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
    result.final || ""
  );

  return result;
}

function saveOptimizedTextFromSidebar(text) {
  PropertiesService.getDocumentProperties().setProperty(
    "latestOptimizedText",
    text || ""
  );

  return {
    success: true,
    message: "Optimized text saved."
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

  if (!rangeElements.length) {
    return {
      success: false,
      message: "No selected text found."
    };
  }

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