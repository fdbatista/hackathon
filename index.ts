import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import csv from 'csv-parser';

import dayjs from 'dayjs';
import customParseFormat from 'dayjs/plugin/customParseFormat';
import { MinMaxScaler } from './src/scaler';
dayjs.extend(customParseFormat);

const lookBack = 96;
const scaler = new MinMaxScaler();
const modelPath = 'data/output/lstm-model';

const loadData = async (): Promise<{ datetime: Date; values: number[] }[]> => {
    const data: { datetime: Date; values: number[] }[] = [];

    return new Promise((resolve, reject) => {
        fs.createReadStream('data/input/energy-data.csv')
            .pipe(csv({ separator: ';' }))
            .on('data', (row) => {
                const dateTimeStr = `${row['Datum']} ${row['Uhrzeit (Von)']}`;
                const dayJs = dayjs(dateTimeStr, 'DD.MM.YYYY HH:mm:ss');

                const datetime = dayJs.toDate();

                const values = Object.keys(row)
                    .filter((key) => key.startsWith('KWH'))
                    .map((key) => parseFloat(row[key].replace(',', '.')));

                data.push({ datetime, values });
            })
            .on('end', () => resolve(data))
            .on('error', (error) => reject(error));
    });
}

const preprocessData = (data: { datetime: Date; values: number[] }[]): { X: number[][][]; y: number[][] } => {
    const series = data.map((entry) => entry.values);

    // Fit the scaler
    scaler.fit(series);

    // Normalize the data
    const normalizedData = scaler.transform(series);

    const X: number[][][] = [];
    const y: number[][] = [];

    for (let i = 0; i < normalizedData.length - lookBack; i++) {
        X.push(normalizedData.slice(i, i + lookBack));
        y.push(normalizedData[i + lookBack]);
    }

    return { X, y };
}

const getModel = async (inputShape: [number, number]): Promise<tf.LayersModel> => {
    if (fs.existsSync(modelPath)) {
        return await tf.loadLayersModel(`${modelPath}/model.json`);
    }

    const model = tf.sequential();

    model.add(tf.layers.gru({
        units: 64,
        inputShape,  // Adjust the shape to your input data
        returnSequences: false
    }));

    model.add(tf.layers.dense({ units: 13 }));

    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    return model;
}

const trainModel = async (model: tf.LayersModel, X: number[][][], y: number[][]): Promise<void> => {
    const XTensor = tf.tensor3d(X);
    const yTensor = tf.tensor2d(y);

    await model.fit(XTensor, yTensor, {
        epochs: 20,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: tf.callbacks.earlyStopping({ patience: 3 }),
    });

    XTensor.dispose();
    yTensor.dispose();

    await model.save(modelPath);
}

const predict = async (model: tf.LayersModel, input: number[][][]): Promise<number[][]> => {
    const inputTensor = tf.tensor3d(input, [input.length, 96, 13]);
    const predictions = model.predict(inputTensor) as tf.Tensor;
    const result = predictions.arraySync() as number[][];

    inputTensor.dispose();
    predictions.dispose();

    return result;
}

const main = async () => {
    const data = await loadData();

    // Split training (2022-2023) and testing (2024)
    const trainData = data.filter((entry) => entry.datetime < new Date('2024-01-01'));
    const testData = data.filter((entry) => entry.datetime >= new Date('2024-01-01'));

    // Preprocess data
    const { X: X_train, y: y_train } = preprocessData(trainData);
    const { X: X_test } = preprocessData(testData);

    // Get or create the model
    const model = await getModel([lookBack, trainData[0].values.length]);

    // Train the model if it hasn't been trained
    if (!fs.existsSync(modelPath)) {
        await trainModel(model, X_train, y_train);
    }

    // Make predictions for the test set
    const predictions = await predict(model, X_test);

    // Denormalize predictions
    const denormalizedPredictions = scaler.inverseTransform(predictions);

    fs.writeFileSync(`data/output/predictions-${new Date().valueOf()}.json`, JSON.stringify(denormalizedPredictions, null, 2));

    // Return results
    return denormalizedPredictions;

}

main()
    .then(() => {
        console.log('Done');
    })
    .catch(err => console.error(err));
