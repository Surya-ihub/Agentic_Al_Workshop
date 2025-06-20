import express from "express";
import { postFeedback, getSkillAnalysis, markAscomplete } from "../controllers/skillAnalyser.controller.js";

const skillAnalyserRouter = express.Router();

skillAnalyserRouter.post("/data/:userId", postFeedback);
skillAnalyserRouter.get("/data/:userId", getSkillAnalysis);
skillAnalyserRouter.post("/mark-module-complete", markAscomplete)

export default skillAnalyserRouter;