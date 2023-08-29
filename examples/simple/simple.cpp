#include <QGuiApplication>
#include <QQmlApplicationEngine>

#include <SpeechToText.h>

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;


    const QUrl url(QStringLiteral("qrc:simple.qml"));

    qmlRegisterType<SpeechToText>("qtwhisper", 1, 0, "SpeechToText");
    qmlRegisterUncreatableType<WhisperInfo>("qtwhisper", 1, 0, "WhisperInfo", "");

    // QML startup
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
      &app, [url](QObject *obj, const QUrl &objUrl){
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);

    engine.load(url);
    return app.exec();
}
