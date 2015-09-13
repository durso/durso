<?php

/**
 * Description of elementFactory
 *
 * @author durso
 */
namespace library\dom\elements\components;


class elementFactory {
    public static function createByTag($tag){
        $element;
        switch($tag){
            case "i":
            case "strong":    
            case "li": 
            case "em":    
            case "b": $element = new inline($tag);
                        break;
            case "label": $element = new label();
                        break;  
            case "hr":
            case "br":
            case "embed":
            case "param":        
            case "meta": $element = new single($tag);
                        break;        
            case "a": $element = new link();
                        break;
            case "img": $element = new img();
                        break;
            case "head": $element = new head();
                        break;
            case "table": $element = new table();
                        break;       
            case "script": $element = new script();
                        break;
            case "input": $element = new input();
                        break;
            case "select": $element = new select();
                        break;        
            case "html": $element = new html();
                        break;  
            case "body": $element = new body();
                        break;
            case "title": $element = new title();
                        break;   
            default: $element = new block($tag);
                        break;
        }
        return $element;
    }
    public static function createText($value){
        return new text($value);
    }
}
