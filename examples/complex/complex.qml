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

  RowLayout {
    id: headerRow
    anchors {
      top: parent.top
    }
    width: parent.width
    height: 200
    ColumnLayout {
      Repeater {
        id: rep
        property var names: ["Normal", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"]
        model: [0, 2, 3, 8, 9, 7]

        RadioDelegate {
          text: rep.names[index]
          enabled: stt.backendInfo.requantizable
                   && (stt.state == SpeechToText.Ready)
          checked: stt.backendInfo.floatType === rep.model[index]
          onClicked: stt.quantize(rep.model[index])
        }
      }
    }
    Item {
      Layout.fillWidth: true
    }

    ColumnLayout {
      Label {
        text: "Collected Info:"
        font.pixelSize: 20
      }
      Label {
        text: stt.backendInfo.floatTypeString
        Layout.leftMargin: 10
        Layout.maximumWidth: 200
        font.pixelSize: 15
        wrapMode: Text.Wrap
      }
      Label {
        text: stt.backendInfo.modelTypeString
        Layout.leftMargin: 10
        font.pixelSize: 15
      }
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
    modelPath: "ggml-tiny.bin"
    onResultReady: function (r) {
      result.text = r
    }
  }
}
