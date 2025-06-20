import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import { dbConnection } from "./database connection/dbConnect.js";
import router from "./routes/user.routes.js";
import skillAnalyserRouter from "./routes/skillAnalyser.routes.js";




dotenv.config();


const app = express();
app.use(express.json())
app.use(cors())

dbConnection()

app.use("/api/user", router)
app.use("/api/skillgap", skillAnalyserRouter)


app.listen(5000, ()=>{
    console.log("Server Connected");
})