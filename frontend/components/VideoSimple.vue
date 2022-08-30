<template>
  <v-container>
    <img ref="video" width="100%">
  </v-container>
</template>

<script>

export default {
  props: {},

  filters: {},

  mounted() {
    setInterval(
      () => {
        fetch('/frame', {method: 'POST'})
        .then(response => response.blob())
        .then(imageBlob => {
          const imageObjectURL = URL.createObjectURL(imageBlob);
          this.$refs.video.src = imageObjectURL
        });
      },
      100
    )
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
