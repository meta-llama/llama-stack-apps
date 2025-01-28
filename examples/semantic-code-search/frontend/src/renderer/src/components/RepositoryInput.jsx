import React, { useState } from 'react'
import PropTypes from 'prop-types'
import { 
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  CircularProgress
} from '@mui/material'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'

const RepositoryInput = ({ onSubmit, isIndexing }) => {
  const [input, setInput] = useState('')
  const [error, setError] = useState('')

  const validateInput = (value) => {
    if (!value) return 'Please enter a GitHub URL or local path'
    
    // Validate GitHub URL
    if (value.startsWith('http')) {
      const githubPattern = /^https:\/\/github\.com\/[\w-]+\/[\w-]+/
      if (!githubPattern.test(value)) {
        return 'Invalid GitHub URL format. Expected: https://github.com/username/repo'
      }
    } else {
      // Validate local path (basic validation)
      if (!value.startsWith('/')) {
        return 'Local path must be absolute (start with /)'
      }
    }
    return ''
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    const validationError = validateInput(input)
    if (validationError) {
      setError(validationError)
      return
    }
    setError('')
    onSubmit(input)
  }

  return (
    <Paper elevation={3} sx={{ p: 4, maxWidth: 600, width: '100%' }}>
      <Typography variant="h5" component="h2" gutterBottom>
        Repository Input
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Enter a GitHub repository URL or select a local repository path to begin.
      </Typography>
      
      <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <TextField
          fullWidth
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter GitHub URL or Local Path..."
          variant="outlined"
          error={!!error}
          helperText={error}
          disabled={isIndexing}
          InputProps={{
            endAdornment: (
              <Button
                variant="contained"
                onClick={() => window.electron.ipcRenderer.invoke('dialog:openDirectory')}
                disabled={isIndexing}
                startIcon={<FolderOpenIcon />}
                sx={{ ml: 1 }}
              >
                Browse
              </Button>
            ),
          }}
        />
        
        <Button
          type="submit"
          variant="contained"
          disabled={!input || isIndexing}
          sx={{ mt: 2 }}
        >
          {isIndexing ? (
            <>
              <CircularProgress size={24} sx={{ mr: 1 }} />
              Indexing Repository...
            </>
          ) : (
            'Index Repository'
          )}
        </Button>
      </Box>
    </Paper>
  )
}

RepositoryInput.propTypes = {
  onSubmit: PropTypes.func.isRequired,
  isIndexing: PropTypes.bool
}

RepositoryInput.defaultProps = {
  isIndexing: false
}

export default RepositoryInput
