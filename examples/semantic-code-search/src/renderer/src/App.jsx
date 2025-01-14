import { useState } from 'react'
import Versions from './components/Versions'
import RepositoryInput from './components/RepositoryInput'
import electronLogo from './assets/electron.svg'

function App() {
  const ipcHandle = () => window.electron.ipcRenderer.send('ping')
  const [isIndexing, setIsIndexing] = useState(false)

  const handleRepositorySubmit = async (repoPath) => {
    setIsIndexing(true)
    try {
      const response = await window.api.indexRepository(repoPath)
      console.log('Repository indexed successfully:', response)
    } catch (error) {
      console.error('Error indexing repository:', error)
      // TODO: Add error handling UI
    } finally {
      setIsIndexing(false)
    }
  }

  return (
    <>
      <div className="container mx-auto p-4">
        <h1 className="text-3xl font-bold text-center mb-8">
          Semantic Code Search
        </h1>
        <RepositoryInput 
          onSubmit={handleRepositorySubmit}
          isIndexing={isIndexing}
        />
      </div>
      <Versions></Versions>
    </>
  )
}

export default App
