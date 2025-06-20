import express from "express";
import { postFeedback, getSkillAnalysis } from "../controllers/skillAnalyser.controller.js";

const skillAnalyserRouter = express.Router();

skillAnalyserRouter.post("/data/:userId", postFeedback);
skillAnalyserRouter.get("/data/:userId", getSkillAnalysis);

export default skillAnalyserRouter;