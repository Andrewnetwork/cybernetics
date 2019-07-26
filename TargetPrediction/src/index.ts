import {Target, MagmaCanvas, Point, Circle, Grid} from "magma-canvas"
import {render, XYPlotData,show, Point2D} from "@tensorflow/tfjs-vis";
import * as TF from "@tensorflow/tfjs";
import * as Tone from "tone";
import { create } from "domain";
import { Tensor } from "@tensorflow/tfjs";

function createModel() : TF.Sequential {
    // Create a sequential model
    const model = TF.sequential(); 
    
    // Add a single hidden layer
    model.add(TF.layers.dense({inputShape: [1], units: 10, useBias: true,activation:"sigmoid"}));
    
    // Add an output layer
    model.add(TF.layers.dense({units: 1, useBias: true }));
  
    return model;
}

/**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 * Source: https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#4
 */
function convertToTensor(data:Point[]) {
    // Wrapping these calculations in a tidy will dispose any 
    // intermediate tensors.
    
    return TF.tidy(() => {
      // Step 1. Shuffle the data    
        TF.util.shuffle(data);
  
        // Step 2. Convert data to Tensor
        const inputs = data.map((point:Point2D) => point.x);
        const labels = data.map((point:Point2D) => point.y);

        const inputTensor = TF.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = TF.tensor2d(labels, [labels.length, 1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();  
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();
  
        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
       
        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });  
}
async function trainModel(model:TF.Sequential, inputs:TF.Tensor<TF.Rank>, labels:TF.Tensor<TF.Rank>){
    // Prepare the model for training.  
    model.compile({
      optimizer: TF.train.adam(),
      loss: TF.losses.meanSquaredError,
      metrics: ['mse'],
    });
    
    const batchSize = 10;
    const epochs = 10;
    
    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'], 
        { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

async function start(){
    let target = new Target({x:400,y:400});
    let motionDat = motionData(target,targetMotion,3000);
    let motionSet : Point[] = [];
    goRight = true;
    let dist = 0;
    for(let i = 0; i < motionDat.length-dist;i++){
        motionSet.push({x:motionDat[i].x,y:motionDat[i+dist].x});
    }

    const model = createModel();
    const {inputs,labels,inputMax,inputMin,labelMax,labelMin} = convertToTensor(motionSet);

    show.modelSummary({name: 'Model Summary'}, model);

    await trainModel(model, inputs, labels);

    const nShots = 10000;
    const delay  = 0;
    const shotFn = modelShotFunction(model,inputMax,inputMin,labelMax,labelMin);

    shootingSimulator(targetMotion,shotFn,{x:400,y:400},nShots,delay,true,false)
        .then((results:SimulationResult)=>simResultDisplay(results,nShots));


    // console.log('Done Training');

    // render.scatterplot(
    //     {name: 'Horsepower v MPG'},
    //     motionSet, 
    //     {
    //     xLabel: 'Horsepower',
    //     yLabel: 'MPG',
    //     height: 300
    //     }
    // );
    // const data = await getData();
    // const values = data.map((d:Car) => ({
    //     x: d.Miles_per_Gallon,
    //     y: d.Horsepower,
    // }));
    //console.log(values);

 
        
    //targetMouseFollow();

    // const data = await getData();
    // const values = data.map((d:Car) => ({
    //     x: d.Miles_per_Gallon,
    //     y: d.Horsepower,
    // }));
    // {values}, 
    //     {
    //     xLabel: 'Horsepower',
    //     yLabel: 'MPG',
    //     height: 300
    //     }
    // );
    // render.scatterplot(
    //     {name: 'Horsepower v MPG'},
    //    
 


    // const canvasDim  = 800;
    // const nShots     = 1000000;
    // const mCanvas    = new MagmaCanvas("canvasContainer",canvasDim,canvasDim,true);
    // const grid       = new Grid(canvasDim,canvasDim);
    // mCanvas.add(grid);
 
    // // let target = new Target({x:400,y:400});
    // // grid.graph((t:number)=>(targetMotion(target,t).x-400)/10);
    // // motionData(target,targetMotion,1000)
    // //     .map((point)=>{return {x:(point.x-400)/4,y:point.y}})
    // //     .map((point,i)=>grid.plot({x:(i-100)/30,y:point.x}));
    // // shootingSimulator(targetMotion,shotFunction,{x:400,y:400},nShots,0,false)
    // // .then((results:SimulationResult)=>simResultDisplay(results,nShots));
    // let target = new Target({x:400,y:400});
    // motionData(target,targetMotion,1000);

}
function modelShotFunction(model:TF.Sequential,inputMax:Tensor,inputMin:Tensor,labelMax:Tensor,labelMin:Tensor){
    return (targetCenter:Point,t:number)=>{
        let prediction : TF.Tensor = (<TF.Tensor>model.predict(TF.tensor2d([targetCenter.x],[1,1]).sub(inputMin).div(inputMax.sub(inputMin)))).mul(labelMax.sub(labelMin)).add(labelMin);
        let predVal = (<Array<Array<number>>>prediction.arraySync())[0][0];
        let predictedShot = {x:predVal,y:targetCenter.y};
        return predictedShot;
    }
}
function motionData(target:Target,targetMotion:Function,maxTime:number){
    let data = [];
    for(let t = 0; t < maxTime; t++){
        targetMotion(target,t);
        data.push(target.center());
    }
    return data;
}
export interface SimulationResult{
    pHit:number;
    score:number;
}
export function simResultDisplay(results:SimulationResult,nShots:number){
    let target = new Target({x:0,y:0});
    console.log(`P(hit)=nHits/nShots=${results.pHit}`);
    console.log(`Area(target)/Area(Ω)=${target.area/(800**2)}`);
    console.log(`Score:${results.score}`);
    console.log(`E[S]=${expectedScore(target,800**2)*nShots}`);
}
// Expected score for a random shooter. 
export function expectedScore(target:Target,omegaArea:number){
    let rings = target.targetRings;
    let e = 0;
    for(let i = rings.length-1; i >= 0; i--){
        let prob = 0;
        if(i == rings.length-1){
            prob = rings[i].area/omegaArea;
        }else{
            prob = (rings[i].area-rings[i+1].area)/omegaArea;
        }
        e += ((i+1)/rings.length)*prob;
    }
    return e;
}



function sound(){
    //play a middle 'C' for the duration of an 8th note
    //frequency.rampTo(3, 0.1);
    
}

export function targetMouseFollow(){
    const canvasDim = 800; 
    const mCanvas   = new MagmaCanvas("canvasContainer",canvasDim,canvasDim,false);
    let target = new Target({x:canvasDim/2,y:canvasDim/2});
    mCanvas.add(target);
    var xTone = new Tone.Oscillator(target.center().x, "sine").toMaster().start();
    var yTone = new Tone.Oscillator(target.center().y, "sawtooth").toMaster().start();
    xTone.volume.value = -15;
    yTone.volume.value = -25;

    mCanvas.addEventListener("mousemove",(ev:MouseEvent,pos:Point)=>{
        
        target.center(pos);
        xTone.frequency.value = target.center().x;
        yTone.frequency.value = target.center().y;
    });

   
}

export function shootingSimulator(targetMotionFn:Function,shotFunction:Function,startingLocation:Point,
                                  nShots:number,delay:number,show:boolean,soundOn:boolean){
    return new Promise<SimulationResult>(function(resolve,reject){
        if(show){
            const canvasDim = 800; 
            const mCanvas   = new MagmaCanvas("canvasContainer",canvasDim,canvasDim,false);
            let target = new Target({x:canvasDim/2,y:canvasDim/2});
            mCanvas.add(target);
            
            if(soundOn){
                var xTone = new Tone.Oscillator(500, "sine").toMaster().start();
                var yTone = new Tone.Oscillator(500, "sawtooth").toMaster().start();
                var xShotTone = new Tone.Oscillator(500, "square").toMaster().start();
                var yShotTone = new Tone.Oscillator(500, "square").toMaster().start();
                xTone.volume.value = -25;
                yTone.volume.value = -40;
                xShotTone.volume.value = -45;
                yShotTone.volume.value = -45;
            }
            
            let t = 0;
            let totalScore = 0;
            let shotCounter = 0;
            let hits = 0;
            let _handler = setInterval(()=>{
                if(shotCounter < nShots){
                    targetMotionFn(target,t);
                    if(soundOn){
                        xTone.frequency.value = target.center().x;
                        yTone.frequency.value = target.center().y;
                    }

                    if(t % 100 == 0){
                        let tmpPoint = {x:target.center().x,y:target.center().y};
                        let shotLoc = shotFunction(tmpPoint,t);
                        if(soundOn){
                            xShotTone.frequency.value = shotLoc.x;
                            yShotTone.frequency.value = shotLoc.y;
                        }
    
                        if(shotLoc != null){
                            let shotScore = target.score(shotLoc);
                            shoot(shotLoc,target,mCanvas,delay/2,true);
                            if(shotScore != 0){
                                hits++;
                                totalScore += shotScore;
                            }
                        }
                        shotCounter++;
                    }
                    t++;
                }else{
                    clearInterval(_handler);
                    resolve({pHit:hits/nShots,score:totalScore});
                    xTone.stop();
                    yTone.stop();
                    xShotTone.stop();
                    yShotTone.stop();
                }
            },0);
    
        }else{
            let target = new Target(startingLocation);
            let t = 0;
            let totalScore = 0;
            let shotCounter = 0;
            let hits = 0;
            while(shotCounter < nShots){
                targetMotionFn(target,t);
                let shotLoc = shotFunction(t);
                if(shotLoc != null){
                    let shotScore = target.score(shotLoc);
                    if(shotScore != 0){
                        hits++;
                        totalScore += shotScore;
                    }
                    shotCounter++;
                }
                t++;
            }
            resolve({pHit:hits/nShots,score:totalScore});
        }
    });
}
function shoot(pos:Point,target:Target,mCanvas:MagmaCanvas,delay=0,ringEffect=true,hitCounter={nHits:0,score:0}){
    let prevHandler : number = null;
    const radius_start = 400;
    const interval_time = 20;
    let radius = radius_start;
    let minRadius = 5;
    let _interval = setInterval(()=>{
        if(prevHandler != null){
            mCanvas.remove(prevHandler);
        }
        radius-=radius_start/(delay/interval_time);
        if( radius <= minRadius){
            //let psn : Point | PointFn = pos;
            let color = "red"
            if(target.contains(pos)){
                // We hit the target. 
                color = "green";
                target.add(new Circle(pos,minRadius,true,color));
                hitCounter.nHits += 1;
                hitCounter.score += target.score(pos);
            }else{
                mCanvas.add(new Circle(pos,minRadius,true,color),0);
            }

            clearInterval(_interval);
        }
        else{
            if(ringEffect){
                prevHandler = mCanvas.add(new Circle(pos,radius,true,"rgba(0,0,0,0.25)"));
            }
        }
    },interval_time);  
}

let goRight = true;
export function targetMotion(target:Target,t:number){
    let rightBoundary = 800-target.radius;
    let leftBoundary = target.radius;

    if(target.center().x > rightBoundary || target.center().x < leftBoundary){
        goRight = !goRight;
    }
    if(goRight){
        target.center({x:target.center().x+1,y:target.center().y});
    }else{
        target.center({x:target.center().x-1,y:target.center().y});
    } 
    return target.center();
}

export function targetMotion2(target:Target,t:number){
    let rightBoundary = 800-target.radius;
    let leftBoundary = target.radius;
  
    if(target.center().x > rightBoundary || target.center().x < leftBoundary){
        goRight = !goRight;
    }
    if(goRight){
        target.center({x:target.center().x+1,y:target.center().y});
    }else{
        target.center({x:target.center().x-1,y:target.center().y});
    } 
    return target.center();
}

export function shotFunction(targetCenter:Point,t:number){
    return targetCenter;
    //return {x:Math.random()*800,y:Math.random()*800};
}
window.addEventListener("DOMContentLoaded",start);