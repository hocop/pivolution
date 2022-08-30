<template>
  <v-container>
    <video ref="video" autoplay="true" width="100%" controls="controls"></video>
  </v-container>
</template>

<script>

export default {
  props: {},

  filters: {},

  methods: {
    handleSendChannelStatusChange (event) {
        if (this.sendChannel) {
            var state = this.sendChannel.readyState;
        }
    },

    async negotiate(pc) {
        pc.addTransceiver('video', {direction: 'recvonly'});
        pc.addTransceiver('audio', {direction: 'recvonly'});

        return pc.createOffer()
        .then(offer => pc.setLocalDescription(offer))
        .then(function() {
            // wait for ICE gathering to complete
            return new Promise(function(resolve) {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    function checkState() {
                        if (pc.iceGatheringState === 'complete') {
                            pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    }
                    pc.addEventListener('icegatheringstatechange', checkState);
                }
            });
        })
        .then(function() {
            var offer = pc.localDescription;
            return fetch('/offer', {
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                }),
                headers: {
                    'Content-Type': 'application/json'
                },
                method: 'POST'
            });
        })
        .then(response => response.json())
        .then(answer => pc.setRemoteDescription(answer))
        .catch(function(e) {
          console.log('Connection error')
            //alert(e);
        });
    },

    connect() {
      var config = {
        sdpSemantics: 'unified-plan'
      };

      this.pc = new RTCPeerConnection(config);

      this.sendChannel = this.pc.createDataChannel("control_channel");
      this.sendChannel.onopen = this.handleSendChannelStatusChange;
      this.sendChannel.onclose = this.handleSendChannelStatusChange;

      const evtListener = (videoRef => evt => {
        console.log('Recieved', evt.track.kind)
        if (evt.track.kind == 'video') {
            videoRef.srcObject = evt.streams[0];
        }
      })(this.$refs.video)
      this.pc.addEventListener('track', evtListener)

      this.negotiate(this.pc);
    }
  },

  mounted() {
    this.connect()
  
    const reconnect_handler = () => {
      console.log(this.pc.iceConnectionState)
      if(this.pc.iceConnectionState == 'disconnected') {
        console.log('Attempting reconnect')
        this.connect()
      }
    }
    setInterval(reconnect_handler, 100)
  },

  destroyed() {
    // close peer connection
    setTimeout(function() {
        this.pc.close();
    }, 500);
  },

  computed: {},

  data() {
    return {
      pc: {},
      sendChannel: {},
    }
  },
}

</script>
