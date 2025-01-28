import { useState } from 'react'
import { Box, Typography } from '@mui/material'

function Versions() {
  const [versions] = useState(window.electron.process.versions)

  return (
    <Box
      sx={{
        display: 'flex',
        gap: 2,
        justifyContent: 'center',
        p: 1,
        backgroundColor: 'transparent',
        borderTop: '1px solid',
        borderColor: 'divider',
        mt: 'auto'
      }}
    >
      <Typography variant="caption" color="text.secondary">
        Electron v{versions.electron}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        Chromium v{versions.chrome}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        Node v{versions.node}
      </Typography>
    </Box>
  )
}

export default Versions
