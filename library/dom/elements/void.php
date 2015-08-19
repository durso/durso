<?php
namespace library\dom\elements;
use library\dom\elements\element;
/**
 * Description of void
 *
 * @author durso
 */
class void extends element{
    /*
     * 
     * Render element to html
     * @return string
     */
    public function render(){
        if($this->isRendered){
            return "";
        }
        $html = $this->openTag();
        return $html;
    }
}
