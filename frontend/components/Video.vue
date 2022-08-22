<template>
  <v-container>
    <audio ref="audio" autoplay="true"></audio>
    <video ref="video" autoplay="true" width="100%" controls="controls"></video>
  </v-container>
</template>

<script>
import { mapGetters, mapActions } from "vuex"

export default {
  props: {},

  filters: {},

  methods: {
    handleSendChannelStatusChange: function (event) {
        if (this.sendChannel) {
            var state = this.sendChannel.readyState;
        }
    },

    negotiate: async function (pc) {
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
            alert(e);
        });
    },
  },

  mounted() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    this.pc = new RTCPeerConnection(config);
    
    this.sendChannel = this.pc.createDataChannel("control_channel");
    this.sendChannel.onopen = this.handleSendChannelStatusChange;
    this.sendChannel.onclose = this.handleSendChannelStatusChange;

    const evtListener = (video_ref => evt => {
      console.log('Recieved', evt.track.kind)
      if (evt.track.kind == 'video') {
          video_ref.srcObject = evt.streams[0];
      }
    })(this.$refs.video)
    this.pc.addEventListener('track', evtListener)

    this.negotiate(this.pc);
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
      pc: null,
      sendChannel: null,
    }
  },
}

</script>
