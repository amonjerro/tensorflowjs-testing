async function getData(){
    const carsDataReq = await fetch('http://localhost:4000/')
    const { data } = await carsDataReq.json()
    const cleaned = data.map(item =>({
        temp:item.temperature,level:item.level,
        volts:item.volts, categ:item.categ
    }))
    return cleaned
}

async function run(){
    const data = await getData()
    let values = data.map(d =>({
        x:d.categ,
        y:d.temp
    }))
    tfvis.render.scatterplot(
        {name:'Category vs. Temperature'},
        {values},
        {
            xLabel:'Category',
            yLabel:'Temperature',
            height:300
        }
    )
    values = data.map(d =>({
        x:d.categ,
        y:d.volts
    }))
    tfvis.render.scatterplot(
        {name:'Category vs. Volts'},
        {values},
        {
            xLabel:'Category',
            yLabel:'Volts',
            height:300
        }
    )
    values = data.map(d =>({
        x:d.level,
        y:d.volts
    }))
    tfvis.render.scatterplot(
        {name:'Level vs. Volts'},
        {values},
        {
            xLabel:'Level',
            yLabel:'Volts',
            height:300
        }
    )
    values = data.map(d =>({
        x:d.level,
        y:d.temp
    }))
    tfvis.render.scatterplot(
        {name:'Level vs. Temperature'},
        {values},
        {
            xLabel:'Level',
            yLabel:'Temperature',
            height:300
        }
    )
    // const model = createModel()
    // tfvis.show.modelSummary({name:'Model Summary'}, model)

    // const tensorData = convertToTensor(data)
    // const {inputs, labels} = tensorData

    // await trainModel(model, inputs, labels)
    // console.log('Done Training')

    // testModel(model, data, tensorData)
}

function createModel(){
    const model = tf.sequential()

    model.add(tf.layers.dense({inputShape:[1], units:1, useBias:true}))
    
    model.add(tf.layers.dense({units:50, activation:'relu', useBias:true}))

    model.add(tf.layers.dense({units:50, activation:'sigmoid', useBias:true}))

    model.add(tf.layers.dense({units:1, activation:'sigmoid',useBias:true}))

    return model
}

function convertToTensor(data){
    return tf.tidy(()=>{
        tf.util.shuffle(data)

        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg)


        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
        const labelTensor = tf.tensor2d(labels, [labels.length, 1])

        const inputMax = inputTensor.max()
        const inputMin = inputTensor.min()

        const labelMax = labelTensor.max()
        const labelMin = labelTensor.min()


        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))


        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
          }
    })
}

async function trainModel(model, inputs, labels){
    model.compile({
        optimizer: tf.train.adam(),
        loss:tf.losses.meanSquaredError,
        metrics:['mse']
    })

    const batchSize = 32
    const epochs = 50

    return await model.fit(inputs, labels,{
        batchSize, epochs,
        callbacks:tfvis.show.fitCallbacks(
            {name:'Training Performance'},
            ['loss','mse'],
            {height:200, callbacks:['onEpochEnd']}
        )
    })
}

function testModel(model, inputData, normalizationData){

    const {inputMax, inputMin, labelMax, labelMin} = normalizationData

    const [xs, preds] = tf.tidy(()=>{
        const xs = tf.linspace(0,1,100)
        const preds = model.predict(xs.reshape([100,1]))
        const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin)

        const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin)

        return [unNormXs.dataSync(), unNormPreds.dataSync()]
    })

    const predictedPoints = Array.from(xs).map((val,i)=>{
        return {x:val, y:preds[i]}
    })

    const originalPoints = inputData.map(d=>({
        x:d.horsepower, y:d.mpg
    }))

    tfvis.render.scatterplot(
        {name:'Model Predictions vs Original Data'},
        {values:[originalPoints, predictedPoints], series:['Original', 'Predictions']},
        {
            xLabel:'Horsepower',
            yLabel:'MPG',
            height:300
        }
    )
}

document.addEventListener('DOMContentLoaded', run)