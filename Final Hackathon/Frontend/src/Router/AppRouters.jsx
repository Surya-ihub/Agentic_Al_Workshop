import React from "react";
import { Route, Routes } from "react-router-dom";
import LoginForm from "../components/Login";
import SignupForm from "../components/Signup";
import Base from "../components/Base/Base";
import SkillSignalUploader from "../pages/Dashboard/Dashboard";

const AppRouters = () => {
  return (
    <div>
      <Routes>
        <Route path="/" element={<LoginForm />} />
        <Route path="/signup" element={<SignupForm />} />
        <Route
          path="/*"
          element={
            <Base>
              <Routes>
                <Route path="/dashboard" element={<SkillSignalUploader />} />
              </Routes>
            </Base>
          }
        ></Route>
      </Routes>
    </div>
  );
};

export default AppRouters;
