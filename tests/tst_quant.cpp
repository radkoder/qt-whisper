#include <QTest>
#include "private/quantization.h"
#include "qbuffer.h"
#include "ggml.h"
#include <QDebug>

class QuantizerTest : public QObject
{
    Q_OBJECT
    const char *base_model_name = "ggml-tiny.bin";
    const char *q40_model_name = "ggml-tiny-q4_0.bin";
    const char *q41_model_name = "ggml-tiny-q4_1.bin";
    const char *q50_model_name = "ggml-tiny-q5_0.bin";
    const char *q51_model_name = "ggml-tiny-q5_1.bin";
    const char *q80_model_name = "ggml-tiny-q8_0.bin";
    ggml_context* _ctx = nullptr;

    void quantize(const char* in, const char* ref_name, ggml_ftype type){

        QFile modelFile{ in };
        modelFile.open(QIODeviceBase::ReadOnly);
        QBuffer result;
        result.open(QIODeviceBase::WriteOnly);

        auto error_code = qtw::buffer_quantize(modelFile, result, type);
        modelFile.close();
        result.close();

        QFile quantized{ ref_name };
        quantized.open(QIODeviceBase::ReadOnly);
        auto ref = quantized.readAll();
        quantized.close();

        QCOMPARE(error_code, 0);
        QCOMPARE(ref.size(), result.buffer().size());
        QCOMPARE(ref.compare(result.buffer()),0);
    }

private slots:

    void initTestCase()
    {
        QVERIFY(QFileInfo{ base_model_name }.size() > 0);
        QVERIFY(QFileInfo{ q40_model_name }.size() > 0);
        QVERIFY(QFileInfo{ q41_model_name }.size() > 0);
        QVERIFY(QFileInfo{ q50_model_name }.size() > 0);
        QVERIFY(QFileInfo{ q51_model_name }.size() > 0);
        QVERIFY(QFileInfo{ q80_model_name }.size() > 0);

        // initializes float-16 lookup table - critical to quantization
        _ctx = ggml_init({});
        QVERIFY(_ctx);

    }
    void q4_0()
    {
        quantize(base_model_name,q40_model_name,GGML_FTYPE_MOSTLY_Q4_0);
    }
    void q4_1()
    {
        quantize(base_model_name,q41_model_name,GGML_FTYPE_MOSTLY_Q4_1);
    }
    void q5_0()
    {
        quantize(base_model_name,q50_model_name,GGML_FTYPE_MOSTLY_Q5_0);
    }
    void q5_1()
    {
        quantize(base_model_name,q51_model_name,GGML_FTYPE_MOSTLY_Q5_1);
    }
    void q8_0()
    {
        quantize(base_model_name,q80_model_name,GGML_FTYPE_MOSTLY_Q8_0);
    }
};

QTEST_MAIN(QuantizerTest)
#include "tst_quant.moc"
