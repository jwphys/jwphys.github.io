# Jekyll plugin to export _data/*.yml as JSON files in assets/i18n/
# This allows the frontend JS to fetch translations dynamically.

module Jekyll
  class I18nGenerator < Generator
    safe true
    priority :low

    def generate(site)
      site.data.each do |lang, data|
        next unless data.is_a?(Hash)
        # Only process root-level YAML files that look like translations
        next unless %w[en zh].include?(lang)

        json_content = JSON.pretty_generate(data)
        dir = File.join(site.dest, 'assets', 'i18n')
        FileUtils.mkdir_p(dir)
        File.write(File.join(dir, "#{lang}.json"), json_content)

        # Also register as a static file so Jekyll copies it properly
        site.static_files << I18nFile.new(site, site.dest, '/assets/i18n', "#{lang}.json")
      end
    end
  end

  class I18nFile < StaticFile
    def write(dest)
      # Already written above
      true
    end
  end
end
