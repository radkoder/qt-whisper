import QtQuick
import QtQuick.Controls.Material
import QtQuick.Layouts
import qtwhisper

ApplicationWindow {
  id: root
  visible: true
  width: 800
  height: 600
  Material.theme: Material.Dark

  ColumnLayout {
    anchors {
      right: parent.right
      top: parent.top
    }
    width: 200

    Label {
      text: "Collected Info:"
      font.pixelSize: 20
    }
    Label {
      text: stt.backendInfo.floatTypeString
      Layout.leftMargin: 10
      font.pixelSize: 15
    }
    Label {
      text: stt.backendInfo.modelTypeString
      Layout.leftMargin: 10
      font.pixelSize: 15
    }
  }

  ColumnLayout {
    anchors.centerIn: parent
    width: parent.width / 2
    height: parent.height / 3

    Item {
      Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
      Layout.fillWidth: true
      Layout.fillHeight: true
      Label {
        id: result

        horizontalAlignment: Text.AlignHCenter
        anchors.fill: parent

        wrapMode: Text.Wrap
        font.capitalization: Font.AllUppercase
        font.pixelSize: 20
        text: "(((Recognised text)))"
        opacity: {
          switch (stt.state) {
          case SpeechToText.Tuning:
          case SpeechToText.WaitingForSpeech:
          case SpeechToText.SpeechDetected:
            return 0.5
          case SpeechToText.Busy:
            return 0.0
          default:
            return 1.0
          }
        }
      }
      BusyIndicator {
        anchors.fill: parent
        running: stt.state === SpeechToText.Busy
      }
    }

    Button {
      text: "Start"
      Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
      enabled: stt.state === SpeechToText.Ready
      onClicked: {
        stt.start()
      }
    }

    Label {
      id: prompt
      Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
      horizontalAlignment: Text.AlignHCenter

      font.pixelSize: 20
      text: {
        switch (stt.state) {
        case SpeechToText.NoModel:
          return "No model loaded"
        case SpeechToText.WaitingForModel:
          return "Loading model, please wait."
        case SpeechToText.Busy:
          return "Inference in progress"
        case SpeechToText.Tuning:
          return "Tuning out background noise"
        case SpeechToText.SpeechDetected:
          return "Speech detected"
        case SpeechToText.WaitingForSpeech:
          return "Speak to start detection"
        case SpeechToText.Ready:
          return "Press start to start listening"
        default:
          return "Unknown state: " + stt.state
        }
      }
    }
  }

  SpeechToText {
    id: stt
    onResultReady: function (r) {
      result.text = r
    }
  }
}
