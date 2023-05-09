const model = tf.sequential()

async function Entrenar() {
    const repeticiones = parseInt(document.getElementById('repeticiones').value);

  
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

   
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  
    await model.fit(xs, ys, {epochs: repeticiones});

    console.log("termino de entrenar mi modelo")

    alert("termino")
}

async function Predecir() {
    const prediccionValor = parseInt(document.getElementById('valorPredecir').value);

    document.getElementById('micro-out-div').innerText =
    model.predict(tf.tensor2d([prediccionValor], [1, 1])).dataSync();
}
