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
    const imgReload = () => {
      fetch('/frame', {method: 'POST'})
      .then(response => response.blob())
      .then(imageBlob => {
          const imageObjectURL = URL.createObjectURL(imageBlob);
          this.$refs.video.src = imageObjectURL
          return imageObjectURL
      })
      .then((imageObjectURL) => {
        URL.revokeObjectURL(imageObjectURL)
        setTimeout(imgReload, 10)
      })
    }

    imgReload()
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
