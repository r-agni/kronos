import WebSocket from 'ws';

async function testWebSocket() {
  console.log('Connecting to ws://localhost:3001/ws ...');
  const ws = new WebSocket('ws://localhost:3001/ws');

  await new Promise<void>((resolve, reject) => {
    ws.on('open', () => {
      console.log('WS Connected successfully!');
      resolve();
    });
    ws.on('error', reject);
  });

  const receivedTypes: string[] = [];

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString());
      if (!receivedTypes.includes(msg.type)) {
        receivedTypes.push(msg.type);
        console.log(`[WS Received Event] ${msg.type}`);
      }
    } catch (e) {}
  });

  // Test 1: TOGGLE_BASELINE
  console.log('Sending TOGGLE_BASELINE...');
  ws.send(JSON.stringify({
    type: 'TOGGLE_BASELINE',
    payload: { enabled: true }
  }));
  await new Promise(r => setTimeout(r, 400));

  // Test 2: VOICE_COMMAND
  console.log('Sending VOICE_COMMAND...');
  ws.send(JSON.stringify({
    type: 'VOICE_COMMAND',
    payload: { transcript: 'Deploy carbon ramp over the chasm', currentPosition: [0, 0, 50] }
  }));
  await new Promise(r => setTimeout(r, 1200));

  // Test 3: SUBMIT_VIEWPORT_FRAME
  console.log('Sending SUBMIT_VIEWPORT_FRAME...');
  ws.send(JSON.stringify({
    type: 'SUBMIT_VIEWPORT_FRAME',
    payload: {
      frameBase64: '',
      telemetry: {
        position: [0, 0, 45],
        speedKmh: 14.2,
        obstacleClearanceMeters: 2.1,
        headingDeg: 5,
        pitchDeg: 1,
        rollDeg: 0,
        traversedDistanceMeters: 45,
        structuralIntegrityPct: 100,
        lidarRayHits: 250,
        autonomousState: 'navigating',
        targetWaypoint: [0, 0, 60]
      }
    }
  }));
  await new Promise(r => setTimeout(r, 1500));

  // Test 4: REQUEST_SECTOR_STREAM
  console.log('Sending REQUEST_SECTOR_STREAM...');
  ws.send(JSON.stringify({
    type: 'REQUEST_SECTOR_STREAM',
    payload: { sectorIndex: 1, currentPosition: [0, 0, 0] }
  }));

  // Wait for stream tokens and sector created
  await new Promise(r => setTimeout(r, 3000));

  console.log('All received event types:', receivedTypes);
  ws.close();
  console.log('WS Test complete.');
  process.exit(0);
}

testWebSocket().catch(err => {
  console.error('WS Test failed:', err);
  process.exit(1);
});
