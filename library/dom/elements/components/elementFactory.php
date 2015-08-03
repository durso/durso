<?php

/**
 * Description of elementFactory
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\components\head;

class elementFactory {
    public static function createByTag($tag){
        $element;
        switch($tag){
            case "div":
            case "address":
            case "article":
            case "aside":
            case "audio":
            case "blockquote":
            case "canvas":
            case "dd":
            case "dl":
            case "fieldset":
            case "figure":
            case "footer":
            case "form":
            case "h1":
            case "h2":
            case "h3":
            case "h4":
            case "h5":
            case "h6":
            case "header":
            case "hgroup":
            case "main":
            case "nav":
            case "noscript":
            case "ol":
            case "output":
            case "pre":
            case "section":
            case "video":   
            case "ul":    
            case "p": $element = new block($tag);
                      break;
            case "i":
            case "li":     
            case "b": $element = new inline($tag);
                        break;
            case "a": $element = new link();
                        break;
            case "img": $element = new img();
                        break;
            case "head": $element = new head();
                        break;
            case "table": $element = new table();
                          break;        
            default: $element = new text("");
        }
        return $element;
    }
}
