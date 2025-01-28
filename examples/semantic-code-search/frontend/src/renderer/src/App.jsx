import { useState } from 'react'
import { Box, Container, Typography, IconButton, CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import ChatPanel from './components/ChatPanel'
import RepositoryInput from './components/RepositoryInput'
import Versions from './components/Versions'
import electronLogo from './assets/electron.svg'

// Create a theme instance
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    background: {
      default: '#f5f5f5',
    },
  },
});

function App() {
  const ipcHandle = () => window.electron.ipcRenderer.send('ping')
  const [isIndexing, setIsIndexing] = useState(false)
  const [isRepositoryIndexed, setIsRepositoryIndexed] = useState(false)

  const handleRepositorySubmit = async (repoPath) => {
    setIsIndexing(true)
    try {
      const response = await window.api.indexRepository(repoPath)
      console.log('Repository indexed successfully:', response)
      setIsRepositoryIndexed(true)
    } catch (error) {
      console.error('Error indexing repository:', error)
      // TODO: Add error handling UI
    } finally {
      setIsIndexing(false)
    }
  }

  const handleBack = () => {
    setIsRepositoryIndexed(false)
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        height: '100vh',
        width: '100vw',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'background.default',
        overflow: 'hidden'
      }}>
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 2,
          position: 'relative',
          px: 2,
          py: 1,
          borderBottom: 1,
          borderColor: 'divider'
        }}>
          {isRepositoryIndexed && (
            <IconButton 
              onClick={handleBack}
              sx={{ 
                '&:hover': {
                  backgroundColor: 'rgba(0, 0, 0, 0.04)'
                }
              }}
            >
              <ArrowBackIcon />
            </IconButton>
          )}
          <Typography 
            variant="h6"
            component="h1" 
            sx={{ 
              textAlign: 'center',
              flex: 1,
              color: 'text.primary'
            }}
          >
            Semantic Code Search
          </Typography>
        </Box>
        
        <Box sx={{ 
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          width: '100%'
        }}>
          {!isRepositoryIndexed ? (
            <Box sx={{ 
              flex: 1, 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              p: 2
            }}>
              <RepositoryInput 
                onSubmit={handleRepositorySubmit}
                isIndexing={isIndexing}
              />
            </Box>
          ) : (
            <ChatPanel />
          )}
        </Box>
        
        <Versions />
      </Box>
    </ThemeProvider>
  )
}

export default App
