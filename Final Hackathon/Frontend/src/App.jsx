import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import AppRouters from './Router/AppRouters'

function App() {
  const [count, setCount] = useState(0)

  return (
   <div>
      <AppRouters/>
   </div>
  )
}

export default App
